from enum import Flag, auto


class ConversationState(Flag):
    IDLE = auto()

    USER_TURN = auto()
    
    ROBOT_PRE = auto()
    '''Processing conversation: can choose tool OR respond/farewell without tool'''
    ROBOT_TOOL_CALLING = auto()
    '''Calling tool'''
    ROBOT_AFT = auto()
    '''Can only generate response/farewell given tool result. (prevent tool loops)'''

    ROBOT_COUNTDOWN = auto()

    CONVERSATION_SHUTDOWN = auto()

    #-----------------------------------------------
    ROBOT_TURN = ROBOT_PRE | ROBOT_TOOL_CALLING | ROBOT_AFT | ROBOT_COUNTDOWN

    IN_CONVERSATION = ROBOT_TURN | USER_TURN

class ExpressionState(Flag):
    IDLE = auto()
    GENERATING = auto()
    TOOL_CALLING = auto()
    SPEAKING = auto()
    LISTENING = auto()
    UNRESPONSIVE_PROCESSING = auto()

class SpeakingEmote(Flag):
    NEUTRAL = auto()
    HAPPY = auto()