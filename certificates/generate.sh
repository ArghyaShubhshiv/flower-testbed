#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Root CA (self-signed)
openssl req -x509 -newkey rsa:4096 -days 3650 -nodes \
  -keyout ca.key -out ca.crt \
  -subj "/C=US/ST=State/L=City/O=FlowerTestbed/CN=FlowerTestbedCA"

# 2. Server key + CSR
openssl genrsa -out server.key 4096
openssl req -new -key server.key -out server.csr -config certificate.conf

# 3. CA signs the server cert, embedding the SAN (server, localhost)
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.pem -days 3650 -extensions v3_req -extfile certificate.conf

rm -f server.csr ca.srl
echo "CREATED : ca.crt, server.pem, server.key"
