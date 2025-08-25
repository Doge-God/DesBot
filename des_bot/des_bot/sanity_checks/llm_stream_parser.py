import asyncio
from typing import AsyncIterable, AsyncIterator, Iterable, List, Optional, Union

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


class SemanticDeltaParser:
    def __init__(self, separators: str = ".!?;:,", min_len = 50):
        self.separators = set(separators)
        self.min_len = min_len
        self.seen = ""   # All processed so far
        self.buffer:str = "" # Unprocessed new text waiting for a boundary

    def parse_new_input(self, new_text: str):
        """
        Given a new input string, return the next semantic delta (sentence/phrase)
        that hasn't been seen yet, or None if nothing new is found.
        Updates internal state to track progress.
        """
        # Find only new text (skip already seen prefix)
        if new_text.startswith(self.seen):
            delta = new_text[len(self.seen):]
            self.seen = new_text
        else:
            # Fallback if stream is inconsistent
            delta = new_text
            self.seen = new_text

        self.buffer += delta

        new_found_sentence_pieces:List[str] = []

        i = 0
        while i < len(self.buffer):
            if self.buffer[i] in self.separators:
                candidate = self.buffer[: i + 1]
                if len(candidate.strip()) >= self.min_len:
                    new_found_sentence_pieces.append(candidate.strip())
                    self.buffer = self.buffer[i + 1 :].lstrip()
                    i = 0
                    continue
            i += 1

        return new_found_sentence_pieces, self.buffer

# Example usage
async def main():
    # async for chunk in stream_tokens("Good morning. Today in the retirement village there will be a tai chi class at 9 and a gardening group at 11. The weather is mild with clear skies and a light breeze. Would you like me to share the lunch menu or this afternoon’s activities?", delay=0.05):
    #     print(chunk)
    mock_llm_stream = stream_tokens("Good morning. Today in the retirement village there will be a tai chi class at 9 and a gardening group at 11. The weather is mild with clear skies and a light breeze. Would you like me to share the lunch menu or this afternoon’s activities? There is also this weird input that's not closed properly", delay=0.05)
    input_parser = SemanticDeltaParser()
    async for new_input in mock_llm_stream:
        sentence_pieces, unclosed_piece = input_parser.parse_new_input(new_input)
        for sentence_piece in sentence_pieces:
            print(f"[{sentence_piece}]")
    print(f"Final: {unclosed_piece}")

if __name__ == "__main__":
    asyncio.run(main())
