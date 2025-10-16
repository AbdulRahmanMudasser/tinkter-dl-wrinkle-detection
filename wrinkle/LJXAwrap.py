# Mock LJXAwrap.py for offline testing
# This file pretends to be the Keyence LJX8_IF API so that code can run without the DLL or hardware.

print("[MOCK] LJXAwrap loaded — no real hardware communication will occur.")

def LJXA_OpenDevice(*args, **kwargs):
    print("[MOCK] LJXA_OpenDevice called")
    return 0  # 0 means success in most Keyence APIs

def LJXA_CloseDevice(*args, **kwargs):
    print("[MOCK] LJXA_CloseDevice called")
    return 0

def LJXA_StartMeasure(*args, **kwargs):
    print("[MOCK] LJXA_StartMeasure called")
    return 0

def LJXA_StopMeasure(*args, **kwargs):
    print("[MOCK] LJXA_StopMeasure called")
    return 0

def LJXA_GetProfile(*args, **kwargs):
    print("[MOCK] LJXA_GetProfile called")
    # Return fake profile data — shape should match what your code expects
    return 0, [0] * 100  # (status, data array)

def LJXA_GetXProfile(*args, **kwargs):
    print("[MOCK] LJXA_GetXProfile called")
    return 0, [0] * 100

def LJXA_GetError(*args, **kwargs):
    print("[MOCK] LJXA_GetError called")
    return 0
