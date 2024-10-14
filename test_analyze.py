import sys
from analyze import main as analyze_main

if __name__ == "__main__":
    test_audio_file = "test-samples/voicekey-test1-human.wav"
    
    # Call the main function from analyze.py
    sys.argv = ["analyze.py", test_audio_file]
    analyze_main()

print("Test completed. If no errors occurred, the analysis module is working correctly.")
