## CHANGELOG

#### 3.9.26 15:30 Utkarsh
Now instead of manually copying the entire dataset and performing a local testtrain split, clients only get their own
shards (which must be created before hand)

#### 3.9.26 03:15 Utkarsh
Communication over gRPC is now actually encrypted

#### 3.9.26 02:24 Utkarsh 
Split into actually creating different client/servers and initializing only what they need 
