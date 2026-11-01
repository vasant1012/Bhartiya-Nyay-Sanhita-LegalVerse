import { useState } from 'react';

import { Header } from './components/Header';
import { Sidebar, type Page } from './components/Sidebar';

import { Explore } from './features/Explore';
import { Placeholder } from './features/Placeholder';
import { SmartSearch } from './features/SmartSearch';
import { SectionHandoff } from './features/SectionHandoff';

import type { Navigate } from './types/app';


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
      'Explore concepts and legal relationships',
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

  const [page, setPage] =
    useState<Page>('explore');

  const [searchQuery, setSearchQuery] =
    useState('');

  const [
    selectedSection,
    setSelectedSection,
  ] = useState('');


  const navigate: Navigate = (
    nextPage,
    options,
  ) => {

    if (options?.query !== undefined) {
      setSearchQuery(options.query);
    }

    if (nextPage !== 'search') {
      setSelectedSection('');
    }

    setPage(nextPage);
  };


  const meta = metadata[page];


  const renderContent = () => {

    if (page === 'explore') {

      return (
        <Explore
          navigate={navigate}
        />
      );
    }


    if (page === 'search') {

      if (selectedSection) {

        return (
          <SectionHandoff
            sectionNumber={
              selectedSection
            }
            onBack={() =>
              setSelectedSection('')
            }
          />
        );
      }

      return (
        <SmartSearch
          initialQuery={searchQuery}
          onOpenSection={
            setSelectedSection
          }
        />
      );
    }


    return (
      <Placeholder
        title={meta.title}
      />
    );
  };


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
        onChange={(nextPage) =>
          navigate(nextPage)
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
          title={meta.title}
          subtitle={meta.subtitle}
        />


        <div
          className="
            min-h-0
            flex-1
            overflow-y-auto
          "
        >

          {renderContent()}

        </div>

      </main>

    </div>
  );
}