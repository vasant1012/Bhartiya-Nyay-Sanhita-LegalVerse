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


const API_BASE = '/api';
const DEFAULT_TIMEOUT_MS = 15_000;


export class ApiError extends Error {
  status: number;
  detail?: unknown;

  constructor(
    message: string,
    status: number,
    detail?: unknown,
  ) {
    super(message);

    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}


async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {

  const controller =
    new AbortController();

  const timeout = window.setTimeout(
    () => controller.abort(),
    DEFAULT_TIMEOUT_MS,
  );


  try {

    const url =
      `${API_BASE}${path}`;


    const response = await fetch(
      url,
      {
        ...init,

        signal:
          controller.signal,

        headers: {
          Accept:
            'application/json',

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


    const contentType =
      response.headers.get(
        'content-type',
      );


    let payload: unknown = null;


    if (
      contentType?.includes(
        'application/json',
      )
    ) {

      payload =
        await response
          .json()
          .catch(() => null);

    } else {

      payload =
        await response
          .text()
          .catch(() => null);
    }


    if (!response.ok) {

      let message =
        `Request failed (${response.status})`;


      if (
        typeof payload === 'object'
        &&
        payload !== null
        &&
        'detail' in payload
      ) {

        const detail =
          (
            payload as {
              detail?: unknown;
            }
          ).detail;


        if (
          typeof detail === 'string'
        ) {
          message = detail;
        }
      }


      throw new ApiError(
        message,
        response.status,
        payload,
      );
    }


    return payload as T;

  } catch (error) {

    if (
      error instanceof ApiError
    ) {
      throw error;
    }


    if (
      error instanceof DOMException
      &&
      error.name === 'AbortError'
    ) {

      throw new ApiError(
        'Backend request timed out.',
        0,
      );
    }


    throw new ApiError(
      error instanceof Error
        ? error.message
        : 'Unable to communicate with LegalVerse backend.',
      0,
      error,
    );

  } finally {

    window.clearTimeout(
      timeout,
    );
  }
}


export const api = {

  health:
    (): Promise<HealthResponse> =>
      request<HealthResponse>(
        '/health',
      ),


  search:
    (
      payload: SearchRequest,
    ): Promise<SearchResponse> =>
      request<SearchResponse>(
        '/search',
        {
          method: 'POST',

          body:
            JSON.stringify(
              payload,
            ),
        },
      ),


  section:
    (
      sectionNumber: string,
    ): Promise<Section> =>
      request<Section>(
        `/sections/${encodeURIComponent(
          sectionNumber,
        )}`,
      ),


  sectionInsights:
    (
      sectionNumber: string,
    ): Promise<SectionInsights> =>
      request<SectionInsights>(
        `/insights/sections/${encodeURIComponent(
          sectionNumber,
        )}`,
      ),

};

