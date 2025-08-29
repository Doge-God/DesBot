from typing import List



class SemanticDeltaParser:
    def __init__(self, separators: str = ".!?;:,", min_len = 20):
        self.separators = set(separators)
        self.min_len = min_len
        self.seen = ""   # All processed so far
        self.buffer:str = "" # Unprocessed new text waiting for a boundary

    def parse_new_input(self, new_text: str):
        """
        Given a new input string, return the next semantic delta (sentence/phrase)
        that hasn't been seen yet (lists[str]), and current buffer for unclosed str.
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

