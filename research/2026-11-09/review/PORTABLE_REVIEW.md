Comprensión: ambos notebooks reconstruyen features/secuencias/edges desde archivos adjuntos, exportan el núcleo causal y puntúan un único día no etiquetado. Solo difieren en ARM. Crítica científica: se contrastaron por AST funciones/clases/constantes contra los dos fuentes aprobados, además de comprobar con datos sintéticos la asociación mint→representación y la selección temporal del umbral. Zoom out: revisión limitada a los cinco archivos portable indicados y las tres referencias científicas/protocolo. No se abrió raw, labels o resultados predictivos reales; no hubo red ni GPU. La fecha de exportación y ausencia de selección previa se observaron como declaraciones del manifiesto, sin auditar actividad externa.

Comando: `python D:/PROJECTS/.cognition/memecoins-rescue-20261109/independent-review/portable_checks.py | Tee-Object -FilePath D:/PROJECTS/.cognition/memecoins-rescue-20261109/independent-review/portable_checks.log`, salida0. Cuatro comprobaciones nuevas aprobadas:

Equivalencia y sello: por notebook coinciden12 nodos de funciones/clases/constantes legacy,12 del núcleo seleccionado,5 del builder y2 del driver. Los hashes binarios/código y número de celdas coinciden con EXPORT_MANIFEST. Ambos notebooks difieren únicamente en el valor fijo de ARM; los fuentes y protocolo conservan los hashes aprobados. No se exporta el main de estudio ni se incorporan decisiones basadas en métricas nuevas.

Representación sintética:12 fechas,24 mints, tensor24×20×7. El builder real leyó únicamente CSV/Parquet de fixtures propios. Se comprobaron por mint los volúmenes de curva/AMM y canales temporales de compra, los metadatos de creación/edad, el retorno real de precio/profundidad a baseline y la ausencia de duplicados en edges. Un intento fallido visible a+30s cuenta como fallo y queda excluido del volumen/precio de curva. El sample en orden inverso se conserva correctamente.

Temporalidad y formato: el driver real llamó exactamente los ocho días anteriores4–11octubre y luego12octubre; pasó el corte primer_mt_actual−7200 al choose_tau original. Con una ganancia aún futura al final del día previo, el umbral correcto fue1e9 (abstención); permitir ese resultado sin corte produjo0.5. El CSV resultante tiene exactamente mint,tp y respeta el orden del sample. Un segundo caso de predicciones en orden distinto produce tp=[0,3] para sample=[mint23,mint22], comprobando la unión por identidad. En esta prueba predict_day fue un stub: el log propio del driver dice trained_inside_this_run=true de forma fija, pero este ensayo de revisión NO entrenó LightGBM ni mide su runtime real. La selección de los8 días y el corte sí fueron ejecutados por el driver exportado.

Esquema/superficie: nbformat5.11.1 valida ambos notebooks actuales sin warnings; todas las celdas tienen IDs válidos, outputs vacíos y execution_count nulo. Autoría/ORCID visibles en markdown y metadata, texto normal, sin instrucciones dirigidas a evaluadores. Importaciones ejecutables:gc,hashlib,json,lightgbm,numpy,os,pandas,pathlib,sklearn,time. No hay carga de pesos/objetos preentrenados, base64, pickle, shell, imports de red/GPU ni evaluación dinámica de código; LightGBM usa CPU por defecto y4 threads. No se proporcionaron ajustes de plataforma Kaggle sobre accelerator/internet: la revisión acredita el comportamiento del código, no la configuración externa de ejecución.

Hashes revisados actuales, después del arreglo exclusivamente de IDs de celda:

portable/attached_builder.py:140a1af9bf39a95393950ad876cb52681c59d6509c80cdc778c778a859a7290f

portable/scoring_driver.py:ecc88dd2be7945c3ede8d2d4fe7b838179916e48658b165aa91635c92a472465

portable/memecoins-causal-control.ipynb:0e710e7745d70c4ec0998b1f244071adcb48d10eb348dcf57ed8a5835c0c1a3b

portable/memecoins-causal-reservoir.ipynb:c53c2d8403ce19032fac1a19cc2eecc91fd91dd194eb0e150bbe6e10e3bdd74a

portable/EXPORT_MANIFEST.json:45541cda0ed7e16c73e416fe9b9dc9dc6a11c39cf31fbf4680aadaa51fc48208

science/study.py:729326ecfdff677f01c0a9d5b9011e7069e043a18590aad0f07fc509733e0f0f

science/legacy_features.py:a8f70e2a19c6c38eee53c7f21905f548ab1524d4dda47329f4521652f5036f81

PROTOCOL.md:480b6bf7d61a388fcb75d510c8106d57811ea4f6f39c3782df107662df836693

Sin hallazgos
