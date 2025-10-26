// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

client<llm> CustomGPT4o {
  provider openai
  options {
    model "gpt-4o"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> CustomGPT4oMini {
  provider openai
  retry_policy Exponential
  options {
    model "gpt-4o-mini"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> GPT41NanoFast {
  provider openai
  retry_policy Exponential
  options {
    model "gpt-4.1-nano"
    api_key env.OPENAI_API_KEY
    service_tier "priority"
  }
}

client<llm> MistralInstruct {
  provider "openai-generic"
  options {
    base_url "http://192.168.137.195:11434/v1"
    model "MistralInstruct16K"
    default_role "system"
    allowed_roles ["user", "assistant", "tool", "system"]
  }
}


// https://docs.boundaryml.com/docs/snippets/clients/retry
retry_policy Constant {
  max_retries 3
  // Strategy is optional
  strategy {
    type constant_delay
    delay_ms 200
  }
}

retry_policy Exponential {
  max_retries 2
  // Strategy is optional
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}