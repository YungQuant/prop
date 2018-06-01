# Yung Quant's prop and shit...

Property of Nonpariel Capital *Employee/Contractor* eyes only...

## Notes from Chris the Cuck

1. Please don't dump your input and output data in the github repo... It bloats the repository and makes it harder to manage.

2. If you do need to dump your data in the repo. Please use `.gitattributes` and `git-lfs`. If you don't know what that is... Google it: https://git-lfs.github.com/... Brownie Points if you store data in .tar.gz format or serialized flat buffers.

## Setup

```
pip install -r requirements.txt
```

## Running and Testing Algos

This needs to be discussed with YungQuant... Cuz' this shit is a mess.


## Accessing Dynamo DB

Add the following to a file at `~/.aws/credentials`

```
[default]
aws_access_key_id = YOUR_KEY
aws_secret_access_key = YOUR_SECRET
```
