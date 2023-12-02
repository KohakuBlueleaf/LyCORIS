import toml


def read_preset(preset):
    try:
        return toml.load(preset)
    except Exception as e:
        print("Error: cannot read preset file. ", e)
        return None
