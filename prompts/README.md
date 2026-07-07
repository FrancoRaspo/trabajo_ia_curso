# Prompts del sistema

Cada archivo `.txt` de esta carpeta es **un prompt** que se le envía al modelo de
lenguaje. La idea es que un experto de crédito pueda **leer y editar** los prompts
sin tocar el código Python.

## Cómo editarlos

- Abrí el `.txt`, cambiá el texto y guardá. El cambio impacta en el **próximo
  informe** (no hace falta reiniciar nada).
- **No borres los `{marcadores}` entre llaves.** El sistema los reemplaza en
  tiempo de ejecución por datos reales (el JSON del cliente, la recomendación ya
  redactada, etc.). Si sacás uno, esa información no llega al modelo.
- No agregues llaves `{ }` sueltas por fuera de los marcadores listados abajo:
  las llaves tienen un significado especial. (Excepción: `juez_sys.txt`, que
  contiene un ejemplo JSON con llaves y se usa tal cual.)
- Evitá que el editor "acomode" o corte las líneas automáticamente. El
  `.editorconfig` de esta carpeta ya lo desactiva para la mayoría de los editores.

## Archivos

### Informe crediticio (cliente encontrado) — se arma por secciones
| Archivo | Para qué sirve | Marcadores |
|---|---|---|
| `secciones_base.txt` | Reglas comunes a TODAS las secciones (tono, qué no inventar, formato de pesos). | — |
| `seccion_human.txt` | Mensaje que le pasa el contexto (JSON) a cada sección. | `{ctx}` |
| `seccion_clasificacion.txt` | Instrucción de la sección **Clasificación de Riesgo**. | — |
| `seccion_historial.txt` | Instrucción de la sección **Historial Crediticio**. | — |
| `seccion_bcra.txt` | Instrucción de la sección **Situación en el Sistema Financiero (BCRA)**. | — |
| `seccion_financiera.txt` | Instrucción de la sección **Información Financiera**. | — |
| `seccion_cumplimiento.txt` | Instrucción de la sección **Cumplimiento de Políticas**. | — |
| `recomendacion.txt` | Instrucción de la sección **Recomendación**. | — |
| `resumen.txt` | Instrucción del **Resumen Ejecutivo**. | — |

> Las secciones `secciones_base.txt` + la instrucción de cada sección se combinan
> automáticamente (base primero, instrucción después).

### Informe cuando el solicitante NO está en la base interna
| Archivo | Para qué sirve | Marcadores |
|---|---|---|
| `sin_datos_sys.txt` | Instrucciones del informe "sin datos internos". | `{regla_bcra}`, `{secciones}` |
| `sin_datos_hum.txt` | Datos del solicitante que se le pasan al modelo. | `{razon_social}`, `{cuit}`, `{tipo_decision}`, `{bcra}` |
| `sin_datos_regla_bcra_con_datos.txt` | Regla cuando SÍ hay datos del BCRA. | — |
| `sin_datos_regla_bcra_sin_datos.txt` | Regla cuando se consultó el BCRA y no había registros. | — |
| `sin_datos_regla_bcra_no_consultado.txt` | Regla cuando no se pudo consultar el BCRA. | — |

### Extractor de la consulta en lenguaje natural
| Archivo | Para qué sirve | Marcadores |
|---|---|---|
| `extractor_sys.txt` | Instrucción para extraer CUIT / razón social / motivo de un texto libre. | — |
| `extractor_hum.txt` | La consulta del usuario. | `{consulta}` |

### Auditor / Juez (evalúa la calidad del informe generado)
| Archivo | Para qué sirve | Marcadores |
|---|---|---|
| `juez_sys.txt` | Rol y criterios del auditor + el formato JSON de la evaluación. | — |
| `juez_hum.txt` | Le pasa los datos fuente y el informe a auditar. | `{datos_fuente}`, `{informe}` |
