class Message {
    role "user" | "assistant" | "tool"
    content string
}

class ReplyTool {
    action_name "reply" @stream.done  @description(#"
        to reply to the user with a message."#)
    response string 
}

class BookActivityTool {
    action_name "book_activity" @stream.done @description(#"
        book an activity for the user. Ensure the user agrees to book this activity, and that the activity is in the schedule.
    "#)
    activity string @description(#"
        name of the activity to book
    "#)
}

class StopTool {
    action_name "stop" @description(#"
        when the user indicate clear intention to stop the conversation
    "#) @stream.done
    response string @description(#"
        indicate clearly to the user that the conversation is stopping
    "#)
}

class SuggestActivityTool {
    action_name "suggest_activity" @stream.done @description(#"
        submit an activity suggestion to Oceania. Ensure the user agrees to sharing the suggestion before using this action
    "#)
    description string 
}

function PrelimAgent(messages: Message[], activities: string) -> (ReplyTool | StopTool | BookActivityTool | SuggestActivityTool) {
    client MistralInstruct //GPT41NanoFast //MistralInstruct //CustomGPT4oMini //"openai/gpt-4o" // Set OPENAI_API_KEY to use this client.
    prompt #"
        You are a helpful assistant for elders in a retirement village. 

        Only reply with given JSON format and never raw strings. 
        "reply" is the default action to respond to user.
        Further information can be fetched with given actions.
        Some actions interact with wider system at Oceania to provide further assistant, but ensure you have all relevant information before using them. If not, use "reply" to ask the user for required information.

        Today's activities are:
        {{activities}}

        {{ ctx.output_format }}

        Messages:
        {{messages}}
    "#
}

function RespondAgent(messages: Message[]) -> (ReplyTool | StopTool) {
    client MistralInstruct //GPT41NanoFast //MistralInstruct //CustomGPT4oMini //"openai/gpt-4o" // Set OPENAI
    prompt #"
        You are a helpful assistant for elders in a retirement village. 

        Only reply with given JSON format and never raw strings. 
        "reply" is the default action to respond to user.

        Messages:
        {{messages}}

        {{ ctx.output_format }}
    "#
}

test ToolInteractionCheck {
    functions [PrelimAgent]
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
            },
            { 
                role "tool"
                content "Sunny. 32 Celsius."
            }
        ]
        activities "Bowling, 10am. Yoga, 2pm. Movie, 6pm."
    }
    @@assert ({{this.action_name == "reply"}})
}

test FurtherInformationCheck {
    functions [PrelimAgent]
    args {
        messages [
            {
                role "user"
                content "Can you book the activity this afternoon for me?"
            }
        ]

        activities "Bowling, 10am. Yoga, 2pm. Movie, 6pm."
    }
    @@assert ({{this.action_name == "reply"}})
}

test InvalidActivityCheck {
    functions [PrelimAgent]
    args {
        messages [
            {
                role "user"
                content "Can you book the recycling workshop this afternoon for me?"
            }
        ]

        activities "Bowling, 10am. Yoga, 2pm. Movie, 6pm."
    }
    @@assert ({{this.action_name == "reply"}})
}

test BookCheck {
    functions [PrelimAgent]
    args {
        messages [
            {
                role "user"
                content "Can you book the yoga class this afternoon for me?"
            },
            {
                role "assistant"
                content "Sure, could I confirm you would like me to book the class for the 2pm session?"
            },
            {
                role "user"
                content "Yes."
            },
        ]

        activities "Bowling, 10am. Yoga, 2pm. Movie, 6pm."
    }
    @@assert ({{this.action_name == "book_activity"}})
}