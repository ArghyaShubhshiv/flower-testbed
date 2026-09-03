### INSTRUCTIONS

Run once:

```
#certificates
chmod +x certificates/generate.sh
./certificates/generate.sh

#create client shards
python3 prepare_data.py

docker-compose build
```

Then continue with normal `docker-compose up` work


