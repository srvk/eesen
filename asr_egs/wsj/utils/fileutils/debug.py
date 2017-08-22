import inspect

def get_debug_info():
  callerframerecord = inspect.stack()[1]
  frame = callerframerecord[0]
  info = inspect.getframeinfo(frame)
  return "file: "+info.filename+" function: "+info.function+" line: "+str(info.lineno)
