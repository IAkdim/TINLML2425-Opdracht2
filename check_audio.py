#!/usr/bin/env python3

import os
import glob

def check_audio_setup():
    """Check audio generation and playback capabilities"""
    print("=== Audio Setup Check ===")
    
    # Check if tomita is available
    try:
        import tomita.legacy.pysynth as ps
        print("✓ tomita library is installed")
    except ImportError:
        print("✗ tomita library not installed")
        print("  Install with: python -m pip install tomita")
        return False
    
    # Check if WAV files exist
    wav_files = glob.glob("*.wav") + glob.glob("music/*.wav")
    if wav_files:
        print(f"✓ Found {len(wav_files)} WAV files:")
        for f in wav_files:
            size = os.path.getsize(f)
            print(f"  {f} ({size} bytes)")
    else:
        print("✗ No WAV files found")
    
    # Test basic music generation
    print("\n=== Testing Music Generation ===")
    try:
        from music.muser import Muser
        
        # Simple test composition
        test_song = (
            [('c', 4), ('d', 4), ('e', 4), ('f', 4)],
            [('c2', 2), ('g2', 2)]
        )
        
        muser = Muser()
        
        # Change to music directory for generation
        original_dir = os.getcwd()
        os.chdir('music')
        
        muser.generate(test_song)
        
        if os.path.exists('song.wav'):
            size = os.path.getsize('song.wav')
            print(f"✓ Successfully generated song.wav ({size} bytes)")
            
            # Move to main directory for easier access
            import shutil
            shutil.move('song.wav', '../test_audio.wav')
            
            os.chdir(original_dir)
            print("✓ Moved to test_audio.wav")
            
        else:
            print("✗ song.wav not generated")
            os.chdir(original_dir)
            
    except Exception as e:
        print(f"✗ Music generation failed: {e}")
        if 'music' in os.getcwd():
            os.chdir('..')
    
    # Audio playback suggestions
    print("\n=== Audio Playback Options ===")
    print("If WAV files are generated but you can't hear them:")
    print("1. Check if audio is muted or volume is low")
    print("2. Try playing with system audio player:")
    print("   - Linux: aplay test_audio.wav")
    print("   - Linux: vlc test_audio.wav") 
    print("   - Or open in file manager")
    print("3. Check if audio drivers are working")
    print("4. Test with headphones if using speakers")
    
    return True

if __name__ == "__main__":
    check_audio_setup()