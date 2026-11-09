Comprensión: la versión revisada compara control contra estados aleatorios de cinco canales, retiene los clasificadores de fit tras calibrar y elige umbrales con resultados disponibles al corte fijo del nuevo día. Crítica científica: las comprobaciones separan exclusión de etiquetas, resolución temporal, fidelidad de la representación y paridad de la optimización. Zoom out: la revisión cubre únicamente PROTOCOL.md y los dos fuentes especificados; no acredita procedencia del cache, adquisición/no inspección de targets reales, rendimiento en servidor ni una ejecución LightGBM completa.

Once comprobaciones distintas ejecutadas y aprobadas; cero hallazgos Mayores abiertos dentro del alcance. Se mantuvieron science y el protocolo en solo lectura. El primer informe REVIEW.md conserva los defectos de V1, ya cerrados por estas comprobaciones de V2.

Comando 1, salida 0: `python D:/PROJECTS/.cognition/memecoins-rescue-20261109/independent-review/fresh_checks_v2.py | Tee-Object -FilePath D:/PROJECTS/.cognition/memecoins-rescue-20261109/independent-review/fresh_checks_v2.log`. Ejecutó diez comprobaciones: invarianza de las features del día frente a sus etiquetas; umbral invariante al resultado aún futuro; secuencia conserva el retorno observado a baseline; secuencia rechaza tiempos -1,60.01,3600 y NaN; main rechaza archivos con mints de secuencia en orden inverso; estados invariantes a etiquetas; calibración temporal; matriz float64 bajo pandas3.1; reputación optimizada contra funciones originales; exclusión efectiva de los canales4,5. El contraejemplo temporal produce ahora tau=1e9 tanto con resultado futuro ganador como perdedor y tau=0.5 cuando el resultado ya está disponible.

Calibración: 3062 filas fit y1082 cal; días22–26 de octubre; mínimo gap observado7211.506849s para fit/cal, train/val y vistas de calibración. Prueba de predict_day con clasificadores stub (42 ajustes), sin workaround de matrix. Se verificó invariancia de las probabilidades y EV frente al cambio de las etiquetas del día actual y supervivencias decrecientes. No se entrenaron clasificadores LightGBM reales.

Paridad de reputación:20 semillas,100 filas por semilla,5 cutoffs,100 comparaciones; error absoluto máximo0. Incluye timestamps empatados, etiquetas ausentes, creadores ausentes, edges vacíos y edges ajenos al panel. La comparación usa creator_rep y wallet_rep preservadas en legacy_features.py, sin leer archivos históricos externos. Cambiar arbitrariamente los canales4,5 del tensor no cambia ni un bit de los96 estados usados por el candidato.

Comando 2, salida0: `python D:/PROJECTS/.cognition/memecoins-rescue-20261109/independent-review/fresh_checks_v2.py check_atomic_predictions | Tee-Object -FilePath D:/PROJECTS/.cognition/memecoins-rescue-20261109/independent-review/atomic_check_v2.log`. Fallo de escritura inyectado después de bytes parciales: archivo final previo conserva su valor7; un archivo final nuevo no aparece; tras escritura correcta se reemplaza y carga valor9. El fallo de escritura se propaga. Los archivos temporales quedan en independent-review.

Hashes comprobados por ambos comandos:

PROTOCOL.md:480b6bf7d61a388fcb75d510c8106d57811ea4f6f39c3782df107662df836693

science/study.py:729326ecfdff677f01c0a9d5b9011e7069e043a18590aad0f07fc509733e0f0f

science/legacy_features.py:a8f70e2a19c6c38eee53c7f21905f548ab1524d4dda47329f4521652f5036f81

Sin hallazgos
