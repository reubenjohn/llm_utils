import json


def pretty_messages(messages: list) -> str:
    result = ""
    for message in messages:
        # print agent name in blue
        result += f"\033[94m{message['sender']}\033[0m: "

        # print response, if any
        if message["content"]:
            result += message["content"] + "\n"
        else:
            result += "\n"

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            result += "\n"
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args))
            result += f"\033[95m{name}\033[0m({arg_str[1:-1]})\n"
    return result


def pretty_print_messages(messages: list):
    print(pretty_messages(messages))
