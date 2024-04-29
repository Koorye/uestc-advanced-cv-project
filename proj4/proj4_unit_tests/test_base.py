def verify(function) -> str:
  """ Will indicate with a print statement whether assertions passed or failed
    within function argument call.
    Args:
    - function: Python function object
    Returns:
    - string
  """
  try:
    function()
    return "\x1b[32m\"Correct\"\x1b[0m"
  except AssertionError:
    return "\x1b[31m\"Wrong\"\x1b[0m"
