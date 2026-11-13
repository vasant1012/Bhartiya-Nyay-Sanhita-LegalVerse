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
  Placeholder,
} from './features/Placeholder';

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
   * Current SPA module.
   */
  const [
    page,
    setPage,
  ] = useState<Page>(
    'explore',
  );


  /*
   * Query handed from Explore
   * into Smart Search.
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
   * Section opened from Chapters.
   *
   * This remains separate from selectedSection
   * so Back returns to the correct parent module.
   */
  const [
    chapterSection,
    setChapterSection,
  ] = useState('');


  /*
   * Central SPA navigation.
   */
  const navigate: Navigate = (
    nextPage,
    options,
  ) => {

    /*
     * Preserve query handed in
     * from Explore.
     */
    if (
      options?.query !== undefined
    ) {

      setSearchQuery(
        options.query,
      );
    }


    /*
     * Clear contextual Search section
     * when leaving Search.
     */
    if (
      nextPage !== 'search'
    ) {

      setSelectedSection('');
    }


    /*
     * Clear contextual Chapter section
     * when leaving Chapters.
     */
    if (
      nextPage !== 'chapters'
    ) {

      setChapterSection('');
    }


    setPage(
      nextPage,
    );
  };


  const meta =
    metadata[page];


  /*
   * -------------------------------------------------------
   * MAIN SPA CONTENT RESOLVER
   * -------------------------------------------------------
   */

  const renderContent = () => {

    /*
     * =====================================================
     * 4.2 EXPLORE
     * =====================================================
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
     * =====================================================
     * 4.3 SMART SEARCH
     * =====================================================
     */
    if (
      page === 'search'
    ) {

      /*
       * ---------------------------------------------------
       * 4.4 SECTION DETAIL
       * opened from Smart Search
       * ---------------------------------------------------
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
          />
        );
      }


      /*
       * Normal Smart Search workspace.
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
     * =====================================================
     * 4.5 CHAPTERS
     * =====================================================
     */
    if (
      page === 'chapters'
    ) {

      /*
       * ---------------------------------------------------
       * 4.4 SECTION DETAIL
       * opened from Chapters
       * ---------------------------------------------------
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
          />
        );
      }


      /*
       * Standard Chapters master/detail view.
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
     * =====================================================
     * 4.6A INSIGHTS - CORPUS OVERVIEW
     * =====================================================
     */
    if (
      page === 'insights'
    ) {

      return (
        <CorpusOverview />
      );
    }


    /*
     * =====================================================
     * FUTURE MODULES
     * =====================================================
     *
     * Bookmarks and About continue to use the common
     * placeholder until Phase 4.7.
     */
    return (
      <Placeholder
        title={
          meta.title
        }
      />
    );
  };


  /*
   * -------------------------------------------------------
   * GLOBAL APPLICATION SHELL
   * -------------------------------------------------------
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
       * Persistent LegalVerse sidebar.
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
       * Application workspace.
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
         * Persistent context header.
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
         * Module workspace.
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