export type Page = 'explore' | 'search' | 'chapters' | 'insights' | 'bookmarks' | 'about';
export type Navigate = (page: Page, options?: { query?: string }) => void;
