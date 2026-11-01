import type {
  SearchRequest,
  SearchResponse,
} from '../types/search';

import type {
  Section,
} from '../types/section';

import type {
  SectionInsights,
} from '../types/insight';

import type {
  ChapterSummary,
  SectionSummary,
} from '../types/chapter';


/*
 * ============================================================
 * Health API Contract
 * ============================================================
 */

export type HealthResponse = {
  status: string;

  version: string;

  availability: {
    source_data: boolean;

    keyword_search: boolean;

    semantic_search: boolean;

    graph_insights: boolean;
  };
};


/*
 * ============================================================
 * API Configuration
 * ============================================================
 *
 * Vite proxies:
 *
 * /api/*
 *
 * to:
 *
 * FastAPI :8000
 *
 * Backend no longer contains the old /v1 prefix.
 */

const API_BASE =
  '/api';


const DEFAULT_TIMEOUT_MS =
  15_000;


/*
 * ============================================================
 * Standard API Error
 * ============================================================
 */

export class ApiError
  extends Error {

  status: number;

  detail?: unknown;


  constructor(
    status: number,
    message: string,
    detail?: unknown,
  ) {

    super(message);

    this.name =
      'ApiError';

    this.status =
      status;

    this.detail =
      detail;
  }
}


/*
 * ============================================================
 * Shared HTTP Request
 * ============================================================
 */

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {

  const controller =
    new AbortController();


  const timeout =
    window.setTimeout(
      () =>
        controller.abort(),
      DEFAULT_TIMEOUT_MS,
    );


  try {

    const response =
      await fetch(
        `${API_BASE}${path}`,
        {
          ...init,

          signal:
            controller.signal,

          headers: {
            Accept:
              'application/json',

            /*
             * Content-Type is only required
             * when sending a request body.
             */
            ...(init?.body
              ? {
                  'Content-Type':
                    'application/json',
                }
              : {}),

            ...(init?.headers ?? {}),
          },
        },
      );


    /*
     * Handle both JSON API responses
     * and unexpected plain-text errors.
     */

    const contentType =
      response.headers.get(
        'content-type',
      );


    const payload: unknown =
      contentType?.includes(
        'application/json',
      )
        ? await response
            .json()
            .catch(
              () => null,
            )

        : await response
            .text()
            .catch(
              () => null,
            );


    /*
     * Normalize backend errors.
     */

    if (
      !response.ok
    ) {

      const detail =
        typeof payload === 'object'
        &&
        payload !== null
        &&
        'detail' in payload

          ? (
              payload as {
                detail?: unknown;
              }
            ).detail

          : undefined;


      const message =
        typeof detail === 'string'

          ? detail

          : `Request failed (${response.status})`;


      throw new ApiError(
        response.status,
        message,
        payload,
      );
    }


    return payload as T;

  } catch (error) {

    /*
     * Preserve normalized API errors.
     */

    if (
      error
      instanceof ApiError
    ) {

      throw error;
    }


    /*
     * Timeout handling.
     */

    if (
      error
      instanceof DOMException
      &&
      error.name ===
        'AbortError'
    ) {

      throw new ApiError(
        0,
        'Backend request timed out.',
      );
    }


    /*
     * Network / Vite proxy / backend
     * connectivity failure.
     */

    throw new ApiError(
      0,

      error instanceof Error

        ? error.message

        : 'Unable to communicate with LegalVerse backend.',

      error,
    );

  } finally {

    window.clearTimeout(
      timeout,
    );
  }
}


/*
 * ============================================================
 * LegalVerse API
 * ============================================================
 */

export const api = {

  /*
   * ----------------------------------------------------------
   * System Health
   *
   * GET /api/health
   * ----------------------------------------------------------
   */

  health:
    (): Promise<HealthResponse> =>

      request<HealthResponse>(
        '/health',
      ),


  /*
   * ----------------------------------------------------------
   * Smart Search
   *
   * POST /api/search
   *
   * Phase 4.3
   * ----------------------------------------------------------
   */

  search:
    (
      payload:
        SearchRequest,
    ): Promise<SearchResponse> =>

      request<SearchResponse>(
        '/search',
        {
          method:
            'POST',

          body:
            JSON.stringify(
              payload,
            ),
        },
      ),


  /*
   * ----------------------------------------------------------
   * Individual Section
   *
   * GET /api/sections/{section_number}
   *
   * Phase 4.4
   * ----------------------------------------------------------
   */

  section:
    (
      sectionNumber:
        string,
    ): Promise<Section> =>

      request<Section>(
        `/sections/${
          encodeURIComponent(
            sectionNumber,
          )
        }`,
      ),


  /*
   * ----------------------------------------------------------
   * Section Smart Insights
   *
   * GET /api/insights/sections/{section_number}
   *
   * Phase 4.4
   * ----------------------------------------------------------
   */

  sectionInsights:
    (
      sectionNumber:
        string,
    ): Promise<SectionInsights> =>

      request<SectionInsights>(
        `/insights/sections/${
          encodeURIComponent(
            sectionNumber,
          )
        }`,
      ),


  /*
   * ----------------------------------------------------------
   * Chapters
   *
   * GET /api/chapters
   *
   * Used by:
   * - Phase 4.5 Chapters
   * - Phase 4.6A Corpus Overview
   * ----------------------------------------------------------
   */

  chapters:
    (): Promise<
      ChapterSummary[]
    > =>

      request<
        ChapterSummary[]
      >(
        '/chapters',
      ),


  /*
   * ----------------------------------------------------------
   * Sections for a Chapter
   *
   * GET /api/sections?chapter_number={chapter}
   *
   * Phase 4.5
   * ----------------------------------------------------------
   */

  sectionsByChapter:
    (
      chapterNumber:
        string,
    ): Promise<
      SectionSummary[]
    > =>

      request<
        SectionSummary[]
      >(
        `/sections?chapter_number=${
          encodeURIComponent(
            chapterNumber,
          )
        }`,
      ),


  /*
   * ----------------------------------------------------------
   * All Sections
   *
   * GET /api/sections
   *
   * Phase 4.6A Corpus Overview
   *
   * Used to calculate the actual source-backed
   * section count. No value is hardcoded.
   * ----------------------------------------------------------
   */

  sections:
    (): Promise<
      SectionSummary[]
    > =>

      request<
        SectionSummary[]
      >(
        '/sections',
      ),

};