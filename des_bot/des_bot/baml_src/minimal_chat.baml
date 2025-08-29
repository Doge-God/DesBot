class Message {
    role "user" | "assistant"
    content string
}

class ReplyTool {
    action_name "reply" @stream.done  @description(#"
        to reply to the user with a message."#)
    response string 
}

class StopTool {
    action_name "stop" @description(#"
        when the user indicate clear intention to stop the conversation
    "#) @stream.done
    response string @description(#"
        indicate clearly to the user that the conversation is stopping
    "#)
}

class QueryPastConversationTool {
    action_name "search_past_conversation" @stream.done @description(#"
        search past conversation with the user to get more information and relevant event.
    "#)
    search_phrases string[] @description(#"
        phrases to search for in the past conversation with user
    "#)
}

//  Messages:
//         {% for message in messages %}
//            {{_.role(message.role)}}
//            {{message.content}}
//         {% endfor %}
//     If there is no proper action to comply, respond with reply and state the limitation clearly.

function MinimalChatAgent(messages: Message[]) -> (ReplyTool | StopTool) {
    client MistralInstruct //CustomGPT4oMini //"openai/gpt-4o" // Set OPENAI_API_KEY to use this client.
    prompt #"
        You are a minimal chat agent. 
        Only reply with given format and never raw strings. 

        {{ ctx.output_format }}

        Messages:
        {{messages}}
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
    @@assert ({{this.action_name == "reply"}})
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
    @@assert ( {{this.action_name == "stop"}})
}

test RobotStartCheck {
    functions [MinimalChatAgent]
    args {
        messages [
            {
                role "assistant"
                content "Good morning. Today's weather is sunny with a high of 25 degrees Celsius. Would you like to hear more about the events?"
            },
            {
                role "user"
                content "No it's fine, but how are you doing today?"
            }
        ]
    }
    @@assert ( {{this.action_name == "reply"}})
}

test AbilityHallucinationCheck {
    functions [MinimalChatAgent]
    args {
        messages [
            {
                role "user"
                content "Give me an outline of today's news."
            }
        ]
    }
    @@assert ( {{this.action_name == "reply"}})
}