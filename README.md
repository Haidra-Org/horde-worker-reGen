# Below are some (possibly) outdated install steps. if you want the latest information. go here
https://discord.com/channels/781145214752129095/1159154031151812650
# horde-worker-reGen
Your memory usage will increase up until the number of queued jobs. Its my belief that you should set your queue size to at least 1, and if you're using threads at least max_threads + 1.
Feel free to try queue size 2 with threads at one and let me know if your kudos/hr goes up or down.
If you have a low amount of system memory (16gb or under), do not attempt a queue size greater than 1 if you have more than one model set to load.
If you plan on running SDXL, you will need to ensure at least 9 gb of system ram remains free.
If you have an 8 gb card, SDXL will only reliably work at max_power values close to 32. 42 was too high for my 2080 in certain cases.
# To run the beta worker (Please read above first)
## (Assuming you have python 3.10 installed):

  - If you did not set `cache_home` before, set it to your old `AI-Horde-Worker`folder for now to avoid redownloading models. IE, `cache_home: "C:/Horde/AI-Horde-Worker/nataili"` (or `models`, depending on when you first installed the worker).
  - If you had previously set `cache_home`, set it to what you were using before.

# Setup venv
- `python -m venv regen`
  - (certain windows setups) `py -3.10 -m venv regen`
- (windows) `regen\Scripts\activate.bat` for cmd or `regen\Scripts\Activate.ps1` for power shell
- (linux) `source regen/bin/activate`
- **Important**: You should now see `(regen)` prepended in your shell. If you do not, try again or ask for help.

# Get worker beta
- `git clone https://github.com/Haidra-Org/horde-worker-reGen.git`
- `cd .\horde-worker-reGen\`
- `pip install -r requirements.txt`

# Run worker
- (Set your config now, adding `SDXL 1.0` if desired)
- `python download_models.py` (To download `SDXL 1.0`) (**critical**)
  - Make sure the folders that are being downloaded to look correct.
- `python run_worker.py` (to start working)
  - Please let me know if you see a variation in your kudos/hr. It does not currently report the models bonus as part of the calculation.

Pressing control-c will have the worker complete any jobs in progress before ending. Please try and avoid hard killing it unless you are seeing many major errors.
