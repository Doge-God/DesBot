import asyncio
from typing import AsyncIterable, AsyncIterator, Iterable, Union

async def stream_tokens(text: str, delay: float = 0.3) -> AsyncIterator[str]:
    """
    Simulate LLM streaming output.

    Args:
        text (str): The full text to stream out.
        delay (float): Delay between token emissions (in seconds).
    
    Yields:
        str: Partial text simulating LLM streaming.
    """
    tokens = text.split()
    current = []
    for token in tokens:
        current.append(token)
        yield " ".join(current)
        await asyncio.sleep(delay)

async def semantic_delta_stream(
    source: AsyncIterable[str],
    separators: str = ".!?;:,",
    min_len: int = 10,
) -> AsyncIterator[str]:
    """
    Yield only new semantic deltas (sections separated by punctuation) from
    a growing text stream (e.g., ab; abc; abcd...)
    """
    # # helper: allow sync iterable
    # async def _aiter_from_iterable(it: Iterable[str]) -> AsyncIterator[str]:
    #     for item in it:
    #         yield item

    ait = source 
    sep_set = set(separators)

    seen = ""   # total processed so far
    buffer = "" # unprocessed new text waiting for a boundary

    async for piece in ait:
        # Find only new text (skip already seen prefix)
        if piece.startswith(seen):
            new_text = piece[len(seen):]
        else:
            # Fallback if stream is inconsistent
            new_text = piece
        seen = piece

        buffer += new_text
        i = 0
        while i < len(buffer):
            if buffer[i] in sep_set:
                candidate = buffer[: i + 1]
                if len(candidate.strip()) >= min_len:
                    yield candidate.strip()
                    buffer = buffer[i + 1 :].lstrip()
                    i = 0
                    continue
            i += 1

    # flush remainder at the end
    if buffer.strip():
        yield buffer.strip()

# Example usage
async def main():
    # async for chunk in stream_tokens("Good morning. Today in the retirement village there will be a tai chi class at 9 and a gardening group at 11. The weather is mild with clear skies and a light breeze. Would you like me to share the lunch menu or this afternoon’s activities?", delay=0.05):
    #     print(chunk)
    mock_llm_stream = stream_tokens("Good morning. Today in the retirement village there will be a tai chi class at 9 and a gardening group at 11. The weather is mild with clear skies and a light breeze. Would you like me to share the lunch menu or this afternoon’s activities?", delay=0.05)
    async for sentence_piece in semantic_delta_stream(mock_llm_stream):
        print(f"[{sentence_piece}]")

if __name__ == "__main__":
    asyncio.run(main())
