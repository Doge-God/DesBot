import rclpy
from rclpy.node import Node
from enum import Enum
from .baml_client import b, partial_types, types
# from .baml_client.stream_types import 
from .baml_client.types import Message
from .baml_client.stream_types import ReplyTool
from dotenv import load_dotenv

load_dotenv()

class ConversationState(Enum):
    NO_CONVERSATION = 0
    LISTENING = 1
    SPEAKING = 2
    THINKING = 3
    SHUTTING_DOWN = 4

class ConversationManagerNode(Node):
    def __init__(self):
        super().__init__('conversation_manager')
        self.state = ConversationState.NO_CONVERSATION
        self.get_logger().info('ConversationManagerNode has been started.')

        self.get_logger().info('Creating a minimal chat agent...')
        chat = [
            Message( role="assistant",content="Hello! How can I assist you today?"),
            Message( role ="user", content="Hmm, what can you do? Give me a detailed list."),
        ]
        stream = b.stream.MinimalChatAgent(chat)

        for partial in stream:
            if isinstance(partial,ReplyTool):
                # print(f"Streamed: {partial}")
                if partial.response:
                    print(partial.response)
        
        # final = stream.get_final_response()
        # final


    def timer_callback(self):
        self.get_logger().info('Timer callback running...')



def main(args=None):
    rclpy.init(args=args)
    node = ConversationManagerNode()
    timer_period = 2.0  # seconds
    node.create_timer(timer_period, node.timer_callback)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()