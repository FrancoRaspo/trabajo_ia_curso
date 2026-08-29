# Flujo de generación en streaming (SSE)

El informe se transmite al navegador **en vivo** mediante *Server-Sent Events*
(SSE). El endpoint `POST /informe/stream` (`app.py`) devuelve un
`StreamingResponse` de `text/event-stream`; cada evento es una línea
`data: {json}` seguida de doble salto de línea.

## Diagrama de secuencia

```mermaid
sequenceDiagram
    autonumber
    participant N as Navegador
    participant A as app.py · FastAPI
    participant P as ReportPipeline
    participant DB as SQL Server
    participant M as XGBoost
    participant R as pgvector + BCRA
    participant L as LLM (por secciones)

    N->>A: POST /informe/stream (consulta + docs)
    A->>P: extraer_datos(consulta)
    P->>L: extractor (CUIT · motivo)
    L-->>P: {cuit, razon_social, tipo_decision}
    A->>A: valida CUIT (dígito verificador)
    A-->>N: etapa: extraccion / documentos

    A->>P: generar_stream(cuit, motivo, razon)
    A-->>N: etapa: sql
    P->>DB: datos, historial, pagos, saldos, mora, features
    DB-->>P: registros del asociado

    alt cliente no encontrado
        P-->>N: informe "sin datos" (bloque único, sin sesión)
    else cliente encontrado
        A-->>N: etapa: ml
        P->>M: predict(features)
        M-->>P: score · nivel · drivers (SHAP)

        A-->>N: etapa: rag
        P->>R: perfil BCRA + recuperar_contexto_completo()
        R-->>P: deuda/situación + políticas, notas, normativa, docs

        P-->>N: meta (score, nivel, badge BCRA)
        A-->>N: etapa: llm

        loop por cada sección
            P->>L: prompt de la sección + contexto acotado
            L-->>N: seccion (título) + token…token (streaming)
        end

        P->>P: validar + guardar log (informes_generados)
        P->>P: retener sesión (sesion_id) para regenerar
        P-->>N: validacion (advertencias)
        P-->>N: fin (informe completo + sesion_id)
    end
```

## Eventos SSE

| `tipo` | Cuándo | Campos |
|---|---|---|
| `etapa` | Al entrar en cada etapa | `clave`: `extraccion` \| `documentos` \| `sql` \| `ml` \| `rag` \| `llm` |
| `meta` | Una vez, antes del informe | `score`, `nivel`, `bcra`, `modelo`, `encontrado`, y (agregados por `app.py`) `numero`, `empresa`, `cuit`, `razon`, `motivo` |
| `seccion` | Al empezar cada sección | `titulo` |
| `token` | Repetido (streaming) | `texto` (incluye el encabezado `## …`) |
| `validacion` | Al terminar | `aprobado`, `advertencias[]` |
| `fin` | Cierre | `informe` (documento en orden de lectura), `sesion_id` |
| `error` | Ante fallo | `mensaje` |

> El **orden de transmisión** es el de ejecución (Clasificación, Historial…,
> Recomendación, Resumen). El evento `fin` trae el documento en **orden de
> lectura** (Resumen arriba), que es el que se usa para el render final y el PDF.

## Render en el navegador

- Durante el stream, los `token` se acumulan y se renderizan como Markdown
  agrupando por *frame* (`requestAnimationFrame`), no por token.
- En `fin`, el documento se parte por encabezados `## ` en **contenedores de
  sección independientes** (`renderInformePorSecciones`). Esto permite, en
  [modo admin](modo-admin.md), regenerar una sola sección in situ.
- El botón **Descargar PDF** re-renderiza el mismo Markdown en el servidor
  (`/api/informe/pdf-render`), así el PDF sale idéntico a lo que se ve.

## Endpoint app-to-app (sin streaming)

`POST /api/informe/pdf` recibe los campos ya estructurados
(`cuit`, `nombre`, `objetivo`), corre el pipeline y devuelve el **PDF** directo.
Reutiliza `generar_stream` internamente, por lo que también cubre el caso "sin
datos internos".
