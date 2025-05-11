# https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683/20

response = openai.Completion.create(
    prompt=raw_text_messages,  # use "completion" techniques
    model=self.model,
    temperature=self.temperature,
    max_tokens=self.max_tokens,  # maximum response length
    stop="\x03",  # Character sequence that terminates output
    top_p=self.top_p,
    presence_penalty=0.0,  # penalties -2.0 - 2.0
    frequency_penalty=0.0,  # frequency = cumulative score
    n=1,
    stream=True,
    logit_bias={"100066": -1},  # example, '～\n\n' token
    user="site_user-id",
    echo=False,
    #logprobs = 5,
)

response = openai.ChatCompletion.create(
        messages    = system + chat[-turns*2:] + user,  # concatenate lists
        # functions   = funct,
        # function_call = "auto",
        model       = model,  # required
        temperature = temperature,
        max_tokens  = max_tokens,  # maximum response length
        stop        = "",
        top_p       = top_p,
        presence_penalty = 0.0,  # penalties -2.0 - 2.0
        frequency_penalty = 0.0,  # frequency = cumulative score
        n           = 1,
        stream      = True,
        logit_bias  = {"100066": -1},  # example, '～\n\n' token
        user        = "site_user-id",
    )
