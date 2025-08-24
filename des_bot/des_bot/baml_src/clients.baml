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

client<llm> MistralInstruct {
  provider "openai-generic"
  options {
    base_url "http://192.168.137.53:11434/v1"
    model "mistral-small3.2:24b-instruct-2506-q4_K_M"
  }
}

client<llm> Mistral {
  provider "openai-generic"
  options {
    base_url "http://192.168.137.53:11434/v1"
    model "mistral-small3.2-16k"
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