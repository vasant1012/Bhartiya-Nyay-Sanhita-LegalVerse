import {
  useState,
} from 'react';

import {
  Header,
} from './components/Header';

import {
  Sidebar,
  type Page,
} from './components/Sidebar';

import {
  Explore,
} from './features/Explore';

import {
  SmartSearch,
} from './features/SmartSearch';

import {
  SectionDetail,
} from './features/section/SectionDetail';

import {
  Chapters,
} from './features/chapters/Chapters';

import {
  CorpusOverview,
} from './features/insights/CorpusOverview';

import {
  Bookmarks,
} from './features/bookmarks/Bookmarks';

import {
  About,
} from './features/about/About';

import {
  useBookmarks,
} from './hooks/useBookmarks';

import type {
  Navigate,
} from './types/app';


const metadata: Record<
  Page,
  {
    title: string;
    subtitle: string;
  }
> = {

  explore: {
    title: 'Explore',
    subtitle:
      'Bharatiya Nyaya Sanhita intelligence workspace',
  },

  search: {
    title: 'Smart Search',
    subtitle:
      'Find relevant BNS sections',
  },

  chapters: {
    title: 'Chapters',
    subtitle:
      'Browse the BNS structure',
  },

  insights: {
    title: 'Insights',
    subtitle:
      'Explore BNS corpus intelligence',
  },

  bookmarks: {
    title: 'Bookmarks',
    subtitle:
      'Saved BNS sections',
  },

  about: {
    title: 'About',
    subtitle:
      'LegalVerse and dataset information',
  },

};


export default function App() {

  /*
   * =========================================================
   * PRIMARY NAVIGATION
   * =========================================================
   */

  const [
    page,
    setPage,
  ] = useState<Page>(
    'explore',
  );


  /*
   * =========================================================
   * SMART SEARCH
   * =========================================================
   */

  const [
    searchQuery,
    setSearchQuery,
  ] = useState('');


  const [
    selectedSection,
    setSelectedSection,
  ] = useState('');


  /*
   * =========================================================
   * CHAPTERS
   * =========================================================
   */

  const [
    chapterSection,
    setChapterSection,
  ] = useState('');


  /*
   * =========================================================
   * BOOKMARKS
   * =========================================================
   *
   * This is the single shared bookmark store used throughout
   * the whole SPA.
   */

  const bookmarkStore =
    useBookmarks();


  /*
   * Section currently opened from Bookmarks.
   */

  const [
    bookmarkSection,
    setBookmarkSection,
  ] = useState('');


  /*
   * =========================================================
   * NAVIGATION
   * =========================================================
   */

  const navigate: Navigate = (
    nextPage,
    options,
  ) => {

    /*
     * Preserve Explore -> Smart Search query.
     */
    if (
      options?.query !== undefined
    ) {

      setSearchQuery(
        options.query,
      );
    }


    /*
     * Clear contextual Smart Search section
     * only when leaving Search.
     */
    if (
      nextPage !== 'search'
    ) {

      setSelectedSection('');
    }


    /*
     * Clear contextual Chapter section
     * only when leaving Chapters.
     */
    if (
      nextPage !== 'chapters'
    ) {

      setChapterSection('');
    }


    /*
     * Clear contextual Bookmark section
     * only when leaving Bookmarks.
     */
    if (
      nextPage !== 'bookmarks'
    ) {

      setBookmarkSection('');
    }


    setPage(
      nextPage,
    );
  };


  const meta =
    metadata[page];


  /*
   * =========================================================
   * CONTENT RESOLVER
   * =========================================================
   */

  const renderContent = () => {

    /*
     * -------------------------------------------------------
     * EXPLORE
     * -------------------------------------------------------
     */

    if (
      page === 'explore'
    ) {

      return (
        <Explore
          navigate={navigate}
        />
      );
    }


    /*
     * -------------------------------------------------------
     * SMART SEARCH
     * -------------------------------------------------------
     */

    if (
      page === 'search'
    ) {

      /*
       * Section Detail opened from Smart Search.
       */
      if (
        selectedSection
      ) {

        return (
          <SectionDetail
            sectionNumber={
              selectedSection
            }

            onBack={() =>
              setSelectedSection('')
            }

            onOpenSection={
              setSelectedSection
            }

            /*
             * Phase 4.7 Bookmark integration.
             */
            bookmarked={
              bookmarkStore
                .isBookmarked(
                  selectedSection,
                )
            }

            onToggleBookmark={
              bookmarkStore
                .toggleBookmark
            }
          />
        );
      }


      return (
        <SmartSearch
          initialQuery={
            searchQuery
          }

          onOpenSection={
            setSelectedSection
          }
        />
      );
    }


    /*
     * -------------------------------------------------------
     * CHAPTERS
     * -------------------------------------------------------
     */

    if (
      page === 'chapters'
    ) {

      /*
       * Section Detail opened from Chapters.
       */
      if (
        chapterSection
      ) {

        return (
          <SectionDetail
            sectionNumber={
              chapterSection
            }

            onBack={() =>
              setChapterSection('')
            }

            onOpenSection={
              setChapterSection
            }

            /*
             * Phase 4.7 Bookmark integration.
             */
            bookmarked={
              bookmarkStore
                .isBookmarked(
                  chapterSection,
                )
            }

            onToggleBookmark={
              bookmarkStore
                .toggleBookmark
            }
          />
        );
      }


      return (
        <Chapters
          onOpenSection={
            setChapterSection
          }
        />
      );
    }


    /*
     * -------------------------------------------------------
     * INSIGHTS
     * -------------------------------------------------------
     */

    if (
      page === 'insights'
    ) {

      return (
        <CorpusOverview />
      );
    }


    /*
     * -------------------------------------------------------
     * BOOKMARKS
     * -------------------------------------------------------
     */

    if (
      page === 'bookmarks'
    ) {

      /*
       * Section Detail opened from Bookmarks.
       */
      if (
        bookmarkSection
      ) {

        return (
          <SectionDetail
            sectionNumber={
              bookmarkSection
            }

            onBack={() =>
              setBookmarkSection('')
            }

            onOpenSection={
              setBookmarkSection
            }

            /*
             * Phase 4.7 Bookmark integration.
             */
            bookmarked={
              bookmarkStore
                .isBookmarked(
                  bookmarkSection,
                )
            }

            onToggleBookmark={
              bookmarkStore
                .toggleBookmark
            }
          />
        );
      }


      /*
       * Main bookmarks workspace.
       */

      return (
        <Bookmarks
          bookmarks={
            bookmarkStore
              .bookmarks
          }

          onOpenSection={
            setBookmarkSection
          }

          onRemove={
            bookmarkStore
              .removeBookmark
          }
        />
      );
    }


    /*
     * -------------------------------------------------------
     * ABOUT
     * -------------------------------------------------------
     */

    if (
      page === 'about'
    ) {

      return (
        <About />
      );
    }


    return null;
  };


  /*
   * =========================================================
   * APPLICATION SHELL
   * =========================================================
   */

  return (

    <div
      className="
        flex
        h-screen
        min-w-[1024px]
        overflow-hidden
        bg-legal-canvas
      "
    >

      <Sidebar
        page={page}

        onChange={(
          nextPage,
        ) =>
          navigate(
            nextPage,
          )
        }
      />


      <main
        className="
          flex
          min-w-0
          flex-1
          flex-col
        "
      >

        <Header
          title={
            meta.title
          }

          subtitle={
            meta.subtitle
          }
        />


        <div
          className="
            min-h-0
            flex-1
            overflow-y-auto
          "
        >

          {
            renderContent()
          }

        </div>

      </main>

    </div>

  );
}