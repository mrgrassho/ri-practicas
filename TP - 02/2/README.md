## Ejercicio 2

#### Dependencias

- Instalar **python**

#### Ejecución

En una terminal correr el siguiente comando:

```bash
python tokenizer_dos.py -h
```
El cual desplegará el `help` del script:

```bash
usage: tokenizer_dos.py [-h] [-n MIN] [-x MAX] [-s STOPWORDS] [-d DIR] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -n MIN, --min MIN     minimun length of terms
  -x MAX, --max MAX     maximun length of terms
  -s STOPWORDS, --stopwords STOPWORDS
                        file containing stopwords
  -d DIR, --dir DIR     directory to scan, default: current working dir

  -v, --verbose         show messages during process
```

#### Ejemplos

Para ejecutar lo solicitado en el **Ejercicio 2**, pegue el siguiente comando en una terminal sobre el siguiente directorio `TP - 02/2`

```bash
python tokenizer_dos.py -s stopwords_spanish.txt -n 3 --dir ../RI-tknz-data
```

La **información solicitada** se almacena en los archivos `frecuencies.json`,  `stats.json` y `terms.json`.
