### Ejecución

Ejecutamos las pruebas utlizando la colección de 4 documentos obtenidos del proyecto Gutenberg y un conjunto de frases utilizadas como queries que se encuentran en `phrase_queries.txt`.

```
./main.py -d ../docs_test/ -q phrase_queries.txt
```

> **Observación**: El script almacena la colección generada en un directorio temporal creado para esa colección, con el objetivo de que podamos luego volver a utilizar el indice generado. El nombre del directorio temporal es obtenido a partir de una operación sobre el path del nombre de la colección.

### Resultados

Los resultados de la prueba los encontramos dentro del directorio generado `<ID>/results/` en los archivos `queries.json`