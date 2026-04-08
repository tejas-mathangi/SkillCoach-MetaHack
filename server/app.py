import uvicorn
import os
import sys

# Ensure the parent directory is in the import path so skillcoach_env can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skillcoach_env import app

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
