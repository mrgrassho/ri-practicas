### Ejecución


Ejecutamos el script de la siguiente manera:

```
./man.py -s ../stopword-list.txt -d ../en/ -t porter -q ../queries_2y3t.txt --weight V1 --metric scalar_prod
```

Para ver mas opciones ejecute el `help` con: 

```
./man.py -h
```

### Pruebas

Corremos el script de prueba utilizando la colección wiki-Small `en/` y el archivo de queries `queries_2y3t.txt`

### Automatizar Pruebas

Utilizamos el script `test.sh` el cual prueba todas las mmetricas con todos los esquemas de weighting disponibles