class GetEventsTool {
    action_name "get_events" @description(#"
        query Oceania for upcoming events for today
    "#) @stream.done
}

class CheckPlantDataTool {
    action_name "check_plant_data" @description(#"
        attempt to use onboard connection to check on soil moisture of
        the plant
    "#) @stream.done
}

class RecallTool {
    action_name "recall" @description(#"
        recall from memory system to better inform the conversation
    "#) @stream.done
    search_phrase string @description(#"
        short description of what information to search for.
    "#) @stream.done
}


function ToolSelectAgent(messages: Message[]) -> GetEventsTool|CheckPlantDataTool|RecallTool|false{
    client GPT41NanoFast //MistralInstruct //CustomGPT4oMini //"openai/gpt-4o" // Set OPENAI_API_KEY to use this client.
    prompt #"
        Determine if the user require using subsquent tools.
        If enough information is present, or no tools are suitable, use false

        {{ messages }}

        {{ ctx.output_format }}
    "#
}

test SanityCheck {
    functions [ToolSelectAgent]
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
    // @@assert (())
}