import { useCallback, useEffect, useState } from 'react';
import type { Bookmark } from '../types/bookmark';

const STORAGE_KEY = 'legalverse.bookmarks.v1';

function readBookmarks(): Bookmark[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed as Bookmark[] : [];
  } catch {
    return [];
  }
}

export function useBookmarks() {
  const [bookmarks, setBookmarks] = useState<Bookmark[]>(readBookmarks);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(bookmarks));
  }, [bookmarks]);

  const isBookmarked = useCallback(
    (sectionNumber: string) => bookmarks.some(item => item.section_number === sectionNumber),
    [bookmarks],
  );

  const addBookmark = useCallback((bookmark: Omit<Bookmark, 'saved_at'>) => {
    setBookmarks(current => {
      if (current.some(item => item.section_number === bookmark.section_number)) return current;
      return [{ ...bookmark, saved_at: new Date().toISOString() }, ...current];
    });
  }, []);

  const removeBookmark = useCallback((sectionNumber: string) => {
    setBookmarks(current => current.filter(item => item.section_number !== sectionNumber));
  }, []);

  const toggleBookmark = useCallback((bookmark: Omit<Bookmark, 'saved_at'>) => {
    setBookmarks(current => {
      const exists = current.some(item => item.section_number === bookmark.section_number);
      if (exists) return current.filter(item => item.section_number !== bookmark.section_number);
      return [{ ...bookmark, saved_at: new Date().toISOString() }, ...current];
    });
  }, []);

  return { bookmarks, isBookmarked, addBookmark, removeBookmark, toggleBookmark };
}
