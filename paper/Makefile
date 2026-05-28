PROJECT ?= 69f58e66fcfdf1cf3536aef1

.PHONY: install figures auth check list pull push sync sync-no-delete compile pdf bbl

install:
	npm install

figures:
	npm run figures:sync

auth:
	npm run overleaf:auth -- --cookie "$$OVERLEAF_SESSION"

check:
	npm run overleaf:check

list:
	npm run overleaf:list

pull:
	npm run overleaf:pull -- "$(PROJECT)" .

push:
	npm run overleaf:push

sync:
	npm run overleaf:sync

sync-no-delete:
	npm run overleaf:sync:no-delete

compile:
	npm run olcli -- compile

pdf:
	npm run overleaf:pdf

bbl:
	npm run overleaf:bbl
