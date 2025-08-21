class Message {
    role string
    content string
}

class ReplyTool {
    response string 
}

class StopTool {
    message string @description(#"
        indicate to the user that the conversation is stopping
    "#)
    action "stop" @description(#"
        when is a good time to stop the conversation
    "#)
}

function MinimalChatAgent(messages: Message[]) -> (ReplyTool | StopTool) {
    client CustomGPT4oMini //"openai/gpt-4o" // Set OPENAI_API_KEY to use this client.
    prompt #"
        You are a minimal chat agent. You will respond to the messages provided.
        If you think the conversation should stop, return a StopTool with a message.
        Otherwise, return a ReplyTool with your response. If there is no proper tool 
        for complying with request, say so clearly

        Messages:
        {{ messages }}

        {{ ctx.output_format }}
    "#
}

test SanityCheck {
    functions [MinimalChatAgent]
    args {
        messages [
            {
                role "user"
                content "Hello, how are you?"
            },
            {
                role "assistant"
                content "I'm doing well, thank you! How can I assist you today?"
            },
            {
                role "user"
                content "What is the weather like today?"
            }
        ]
    }
}

test StopCheck {
    functions [MinimalChatAgent]
    args {
        messages [
            {
                role "user"
                content "Hello, how are you?"
            },
            {
                role "assistant"
                content "I'm doing well, thank you! How can I assist you today?"
            },
            {
                role "user"
                content "Nothing, bye"
            }
        ]
    }
}