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
   * SMART SEARCH STATE
   * =========================================================
   */

  const [
    searchQuery,
    setSearchQuery,
  ] = useState('');


  /*
   * Section opened from Smart Search.
   */
  const [
    selectedSection,
    setSelectedSection,
  ] = useState('');


  /*
   * =========================================================
   * CHAPTER STATE
   * =========================================================
   *
   * Section opened from Chapters.
   */

  const [
    chapterSection,
    setChapterSection,
  ] = useState('');


  /*
   * =========================================================
   * BOOKMARK STATE
   * =========================================================
   */

  const bookmarkStore =
    useBookmarks();


  /*
   * Section currently opened from
   * the Bookmarks tab.
   */
  const [
    bookmarkSection,
    setBookmarkSection,
  ] = useState('');


  /*
   * =========================================================
   * CENTRAL SPA NAVIGATION
   * =========================================================
   */

  const navigate: Navigate = (
    nextPage,
    options,
  ) => {

    /*
     * Preserve a query handed from
     * Explore into Smart Search.
     */
    if (
      options?.query !== undefined
    ) {

      setSearchQuery(
        options.query,
      );
    }


    /*
     * Leaving Smart Search closes any
     * contextual section opened from Search.
     */
    if (
      nextPage !== 'search'
    ) {

      setSelectedSection('');
    }


    /*
     * Leaving Chapters closes any
     * contextual section opened from Chapters.
     */
    if (
      nextPage !== 'chapters'
    ) {

      setChapterSection('');
    }


    /*
     * Leaving Bookmarks closes any
     * contextual section opened from Bookmarks.
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
     * 4.2 EXPLORE
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
     * 4.3 SMART SEARCH
     * -------------------------------------------------------
     */

    if (
      page === 'search'
    ) {

      /*
       * 4.4 Section Detail opened from
       * Smart Search.
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


      /*
       * Standard Smart Search.
       */
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
     * 4.5 CHAPTERS
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


      /*
       * Standard Chapters workspace.
       */
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
     * 4.6A INSIGHTS / CORPUS OVERVIEW
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
     * 4.7 BOOKMARKS
     * -------------------------------------------------------
     */

    if (
      page === 'bookmarks'
    ) {

      /*
       * Existing 4.4 Section Detail reused
       * for sections opened from Bookmarks.
       */
      if (
        bookmarkSection
      ) {

        return (
          <SectionDetail
            sectionNumber={
              bookmarkSection
            }

            /*
             * Back returns to Bookmarks.
             */
            onBack={() =>
              setBookmarkSection('')
            }

            /*
             * Related/cross-reference exploration
             * remains inside the bookmark context.
             */
            onOpenSection={
              setBookmarkSection
            }

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
     * 4.7 ABOUT
     * -------------------------------------------------------
     */

    if (
      page === 'about'
    ) {

      return (
        <About />
      );
    }


    /*
     * This should not normally be reached because
     * all Page values now have an implementation.
     */

    return null;
  };


  /*
   * =========================================================
   * GLOBAL LEGALVERSE SHELL
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

      {/*
       * Persistent navigation.
       */}
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


      {/*
       * Main research workspace.
       */}
      <main
        className="
          flex
          min-w-0
          flex-1
          flex-col
        "
      >

        {/*
         * Context-aware header.
         */}
        <Header
          title={
            meta.title
          }

          subtitle={
            meta.subtitle
          }
        />


        {/*
         * Current module.
         */}
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