### INSTRUCTIONS

Run once:

```
#certificates
chmod +x certificates/generate.sh
./certificates/generate.sh

# 1. Go to the project
cd <project path>

# 2. Install Python virtual-environment support
sudo apt update
sudo apt install python3-venv -y

# 3. Create a virtual environment
python3 -m venv .venv

# 4. Activate it
source .venv/bin/activate

# 5. Upgrade pip
python -m pip install --upgrade pip

# 6. Install packages needed to prepare the data
python -m pip install numpy scikit-learn

# 7. Generate the client datasets
python prepare_data.py

# 8. Stop any old Docker containers for this project
docker compose down

# 9. Build the server and client Docker images
docker compose build

# 10. Start the Flower server and both clients
docker compose up
```

Then continue with normal `docker-compose up` work


