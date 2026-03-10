def validate_int(v: str) -> bool | str:
    return v.lstrip("-").isdigit() or "Enter a valid integer"

def validate_float(v: str) -> bool | str:
    return v.replace(".", "", 1).lstrip("-").isdigit() or "Enter a valid number"
