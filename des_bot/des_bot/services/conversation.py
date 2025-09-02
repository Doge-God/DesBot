import asyncio

class Conversation:
    async def __init__(self, loop:asyncio.AbstractEventLoop):
        self.loop = loop
    
        self.mic_stream = None
        self.speaker_stream = None

    
    async def start_up(self):
        pass
    