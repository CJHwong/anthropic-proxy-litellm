import json
import logging
import os
import random
import string
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use environment variables to point to your custom OpenAI-compatible API
# Example: export OPENAI_API_BASE="http://localhost:8080/v1"
# Example: export OPENAI_API_KEY="your-custom-key"
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Check if the required environment variable for the API base URL is set
if not OPENAI_API_BASE:
    raise ValueError(
        "The 'OPENAI_API_BASE' environment variable is not set. Please specify the URL of your OpenAI-compatible API."
    )

IS_DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

DEFAULT_MODEL = os.getenv('MODEL') or 'my-model'
REASONING_MODEL = os.getenv('REASONING_MODEL') or DEFAULT_MODEL
COMPLETION_MODEL = os.getenv('COMPLETION_MODEL') or DEFAULT_MODEL

app = FastAPI()

# --- Helper Functions ---
# (Helper functions like debug_log, map_stop_reason, etc. remain the same)


def debug_log(*args):
    """Prints debug messages if DEBUG is enabled."""
    if IS_DEBUG:
        logger.info(' '.join(map(str, args)))


def map_stop_reason(finish_reason: str | None) -> str:
    """Maps OpenAI's finish_reason to Anthropic's stop_reason."""
    if not finish_reason:
        return 'end_turn'
    return {
        'tool_calls': 'tool_use',
        'stop': 'end_turn',
        'length': 'max_tokens',
    }.get(finish_reason, 'end_turn')


def normalize_content(content: Any) -> str | None:
    """Normalizes message content to a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join(item.get('text', '') for item in content if item.get('type') == 'text')
    return None


def remove_uri_format(schema: Any) -> Any:
    """Recursively removes 'format: uri' from a JSON schema."""
    if isinstance(schema, dict):
        if schema.get('type') == 'string' and schema.get('format') == 'uri':
            return {k: v for k, v in schema.items() if k != 'format'}
        return {key: remove_uri_format(value) for key, value in schema.items()}
    if isinstance(schema, list):
        return [remove_uri_format(item) for item in schema]
    return schema


def generate_message_id(prefix='msg'):
    """Generates a random message ID."""
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=24))
    return f'{prefix}_{random_part}'


@app.post('/v1/messages')
async def messages_proxy(request: Request):
    try:
        payload = await request.json()
        is_stream = payload.get('stream', False)

        # 1. Translate Anthropic request to OpenAI format
        # This part of the logic remains unchanged as it prepares a standard OpenAI payload
        messages = []
        if 'system' in payload:
            system_content = normalize_content(payload['system'])
            if system_content:
                messages.append({'role': 'system', 'content': system_content})

        for msg in payload.get('messages', []):
            role = msg.get('role')
            content = msg.get('content')
            normalized_text = normalize_content(content)
            tool_calls = []
            if isinstance(content, list):
                tool_calls = [
                    {
                        'type': 'function',
                        'id': item['id'],
                        'function': {'name': item['name'], 'arguments': json.dumps(item['input'])},
                    }
                    for item in content
                    if item.get('type') == 'tool_use'
                ]

            new_msg = {'role': role}
            if normalized_text:
                new_msg['content'] = normalized_text
            if tool_calls:
                new_msg['tool_calls'] = tool_calls

            if new_msg.get('content') or new_msg.get('tool_calls'):
                messages.append(new_msg)

            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'tool_result':
                        messages.append(
                            {
                                'role': 'tool',
                                'content': item.get('content', ''),
                                'tool_call_id': item.get('tool_use_id'),
                            }
                        )

        tools = [
            {
                'type': 'function',
                'function': {
                    'name': tool['name'],
                    'description': tool['description'],
                    'parameters': remove_uri_format(tool['input_schema']),
                },
            }
            for tool in payload.get('tools', [])
        ]

        openai_payload = {
            'model': REASONING_MODEL if payload.get('thinking') else COMPLETION_MODEL,
            'messages': messages,
            'max_tokens': payload.get('max_tokens', 4096),
            'temperature': payload.get('temperature', 1.0),
            'stream': is_stream,
        }
        if tools:
            openai_payload['tools'] = tools

        debug_log('OpenAI Payload:', json.dumps(openai_payload, indent=2))

        # 2. Forward request to your custom OpenAI-compatible API
        headers = {'Content-Type': 'application/json'}
        if OPENAI_API_KEY:
            headers['Authorization'] = f'Bearer {OPENAI_API_KEY}'

        # The URL for the target service is now built from OPENAI_API_BASE
        api_url = f'{OPENAI_API_BASE}/chat/completions'

        if not is_stream:
            return await handle_non_stream(api_url, openai_payload, headers)
        else:
            return StreamingResponse(stream_generator(api_url, openai_payload, headers), media_type='text/event-stream')

    except Exception as e:
        logger.error(f'An error occurred: {e}', exc_info=True)
        return Response(content=json.dumps({'error': str(e)}), status_code=500, media_type='application/json')


async def handle_non_stream(url: str, payload: dict, headers: dict):
    """Handles the non-streaming API response."""
    # This function logic remains the same
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    debug_log('Custom API Response:', json.dumps(data, indent=2))

    choice = data['choices'][0]
    openai_message = choice['message']

    content_parts = []
    if openai_message.get('content'):
        content_parts.append({'type': 'text', 'text': openai_message['content']})

    for tool_call in openai_message.get('tool_calls', []) or []:
        content_parts.append(
            {
                'type': 'tool_use',
                'id': tool_call['id'],
                'name': tool_call['function']['name'],
                'input': json.loads(tool_call['function']['arguments']),
            }
        )

    anthropic_response = {
        'id': data.get('id', '').replace('chatcmpl', 'msg') or generate_message_id(),
        'type': 'message',
        'role': openai_message.get('role', 'assistant'),
        'model': payload['model'],
        'content': content_parts,
        'stop_reason': map_stop_reason(choice.get('finish_reason')),
        'stop_sequence': None,
        'usage': {
            'input_tokens': data['usage']['prompt_tokens'],
            'output_tokens': data['usage']['completion_tokens'],
        },
    }

    return Response(content=json.dumps(anthropic_response), media_type='application/json')


async def stream_generator(url: str, payload: dict, headers: dict):
    """Handles the streaming API response and translates SSE events."""
    # This function logic remains the same
    message_id = generate_message_id()

    def sse_pack(event: str, data: dict) -> str:
        return f'event: {event}\ndata: {json.dumps(data)}\n\n'

    yield sse_pack(
        'message_start',
        {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': payload['model'],
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': 0, 'output_tokens': 0},
            },
        },
    )
    yield sse_pack('ping', {'type': 'ping'})

    text_block_started = False
    tool_call_accumulators: dict[int, dict] = {}
    usage = {}
    last_chunk = {}

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream('POST', url, json=payload, headers=headers, timeout=300) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith('data:'):
                    continue

                data_str = line[len('data: ') :].strip()
                if data_str == '[DONE]':
                    break

                try:
                    chunk = json.loads(data_str)
                    last_chunk = chunk
                except json.JSONDecodeError:
                    debug_log('Failed to parse JSON chunk:', data_str)
                    continue

                if chunk.get('usage'):
                    usage = chunk['usage']

                delta = chunk.get('choices', [{}])[0].get('delta', {})
                if not delta:
                    continue

                if delta.get('content'):
                    if not text_block_started:
                        yield sse_pack(
                            'content_block_start',
                            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                        )
                        text_block_started = True
                    yield sse_pack(
                        'content_block_delta',
                        {
                            'type': 'content_block_delta',
                            'index': 0,
                            'delta': {'type': 'text_delta', 'text': delta['content']},
                        },
                    )

                if delta.get('tool_calls'):
                    for tool_call_delta in delta['tool_calls']:
                        idx = tool_call_delta['index']
                        if idx not in tool_call_accumulators:
                            tool_call_accumulators[idx] = {'id': '', 'name': '', 'args': ''}
                            func_info = tool_call_delta.get('function', {})
                            tool_call_accumulators[idx]['id'] = tool_call_delta.get('id', '')
                            tool_call_accumulators[idx]['name'] = func_info.get('name', '')
                            yield sse_pack(
                                'content_block_start',
                                {
                                    'type': 'content_block_start',
                                    'index': idx,
                                    'content_block': {
                                        'type': 'tool_use',
                                        'id': tool_call_accumulators[idx]['id'],
                                        'name': tool_call_accumulators[idx]['name'],
                                        'input': {},
                                    },
                                },
                            )

                        arg_chunk = tool_call_delta.get('function', {}).get('arguments', '')
                        if arg_chunk:
                            tool_call_accumulators[idx]['args'] += arg_chunk
                            yield sse_pack(
                                'content_block_delta',
                                {
                                    'type': 'content_block_delta',
                                    'index': idx,
                                    'delta': {'type': 'input_json_delta', 'partial_json': arg_chunk},
                                },
                            )

    finish_reason = last_chunk.get('choices', [{}])[0].get('finish_reason')
    stop_reason = map_stop_reason(finish_reason)

    if tool_call_accumulators:
        for idx in tool_call_accumulators:
            yield sse_pack('content_block_stop', {'type': 'content_block_stop', 'index': idx})
    elif text_block_started:
        yield sse_pack('content_block_stop', {'type': 'content_block_stop', 'index': 0})

    yield sse_pack(
        'message_delta',
        {
            'type': 'message_delta',
            'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
            'usage': {'output_tokens': usage.get('completion_tokens', 0)},
        },
    )
    yield sse_pack('message_stop', {'type': 'message_stop'})


if __name__ == '__main__':
    import uvicorn

    port = int(os.getenv('PORT', 3000))
    uvicorn.run(app, host='0.0.0.0', port=port)
