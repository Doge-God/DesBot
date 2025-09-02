class Action{
    
}

function ToolSelect(transcription: string) -> bool {
    client MistralInstruct //CustomGPT4oMini //"openai/gpt-4o" // Set OPENAI_API_KEY to use this client.
    prompt #"
        Is this a complete user message?

        {{ transcription }}

        {{ ctx.output_format }}
    "#
}