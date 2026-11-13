import type {
  ReactNode,
} from 'react';

import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  BrainCircuit,
  ExternalLink,
  Network,
  RefreshCw,
  Tags,
} from 'lucide-react';

import {
  useSectionDetail,
} from './useSectionDetail';


type SectionDetailProps = {
  sectionNumber: string;

  onBack: () => void;

  onOpenSection: (
    sectionNumber: string,
  ) => void;
};


export function SectionDetail({
  sectionNumber,
  onBack,
  onOpenSection,
}: SectionDetailProps) {

  const state =
    useSectionDetail(
      sectionNumber,
    );


  /*
   * Source loading state.
   *
   * Insights are intentionally not allowed to determine
   * whether the statutory source itself can render.
   */
  if (state.sectionLoading) {

    return (
      <div className="p-7">

        <div
          className="
            h-[540px]
            animate-pulse
            rounded-2xl
            border
            border-slate-200
            bg-white
          "
        />

      </div>
    );
  }


  /*
   * Source failure state.
   */
  if (
    state.sectionError
    ||
    !state.section
  ) {

    return (
      <div className="p-7">

        <div
          className="
            rounded-2xl
            border
            border-red-200
            bg-red-50
            p-7
          "
        >

          <button
            onClick={onBack}
            className="
              mb-6
              flex
              items-center
              gap-2
              text-sm
              font-semibold
              text-legal-navy
              hover:underline
            "
          >
            <ArrowLeft size={16} />

            Back to search results
          </button>


          <h2
            className="
              text-xl
              font-bold
              text-red-800
            "
          >
            Section could not be loaded
          </h2>


          <p
            className="
              mt-2
              text-sm
              text-red-700
            "
          >
            {state.sectionError}
          </p>


          <button
            onClick={() =>
              void state.retrySection()
            }
            className="
              mt-5
              flex
              items-center
              gap-2
              rounded-lg
              bg-white
              px-4
              py-2
              text-sm
              font-semibold
              text-red-700
              ring-1
              ring-red-200
            "
          >
            <RefreshCw size={15} />

            Try again
          </button>

        </div>

      </div>
    );
  }


  const section =
    state.section;

  const insights =
    state.insights;


  return (
    <div className="p-7">

      {/* Back navigation */}

      <button
        onClick={onBack}
        className="
          mb-4
          flex
          items-center
          gap-2
          text-sm
          font-semibold
          text-legal-navy
          hover:underline
        "
      >
        <ArrowLeft size={16} />

        Back to search results
      </button>


      {/*
       * Desktop research workspace.
       *
       * Context:       205px
       * Source:        flexible
       * Smart Insight: 300px
       */}

      <div
        className="
          grid
          min-h-[620px]
          grid-cols-[205px_minmax(0,1fr)_300px]
          overflow-hidden
          rounded-2xl
          border
          border-legal-border
          bg-white
          shadow-sm
        "
      >

        {/* =====================================================
            LEFT: SECTION CONTEXT
        ====================================================== */}

        <aside
          className="
            border-r
            border-legal-border
            bg-slate-50
            p-5
          "
        >

          <div
            className="
              text-[11px]
              font-bold
              uppercase
              tracking-wider
              text-legal-muted
            "
          >
            Section Context
          </div>


          <div
            className="
              mt-5
              rounded-xl
              bg-white
              p-4
              ring-1
              ring-slate-200
            "
          >

            <div
              className="
                text-[11px]
                font-semibold
                text-legal-gold
              "
            >
              CHAPTER {section.chapter_number}
            </div>


            <div
              className="
                mt-2
                text-[13px]
                font-bold
                leading-5
                text-legal-text
              "
            >
              {section.chapter_name}
            </div>


            {section.chapter_subtype && (

              <div
                className="
                  mt-3
                  rounded-lg
                  bg-slate-100
                  px-3
                  py-2
                  text-[11px]
                  leading-4
                  text-legal-muted
                "
              >
                {section.chapter_subtype}
              </div>

            )}

          </div>


          <div
            className="
              mt-6
              text-[11px]
              leading-5
              text-legal-muted
            "
          >
            Original statutory content is
            displayed without ML rewriting.
          </div>

        </aside>


        {/* =====================================================
            CENTER: AUTHORITATIVE STATUTORY SOURCE
        ====================================================== */}

        <main
          className="
            min-w-0
            px-7
            py-6
          "
        >

          <div
            className="
              flex
              items-center
              gap-2
              text-[11px]
              font-bold
              uppercase
              tracking-widest
              text-legal-gold
            "
          >
            <BookOpen size={14} />

            Section {section.section_number}
          </div>


          <h2
            className="
              mt-3
              max-w-3xl
              text-[28px]
              font-bold
              leading-[1.18]
              text-legal-text
            "
          >
            {section.section_name}
          </h2>


          <div
            className="
              mt-4
              flex
              flex-wrap
              gap-2
              text-[11px]
              text-legal-muted
            "
          >

            <span
              className="
                rounded-full
                bg-slate-100
                px-3
                py-1.5
              "
            >
              Chapter {section.chapter_number}
            </span>


            {section.chapter_subtype && (

              <span
                className="
                  rounded-full
                  bg-slate-100
                  px-3
                  py-1.5
                "
              >
                {section.chapter_subtype}
              </span>

            )}

          </div>


          <div
            className="
              my-6
              h-px
              bg-slate-200
            "
          />


          <div
            className="
              text-[11px]
              font-bold
              uppercase
              tracking-wider
              text-legal-muted
            "
          >
            Provision
          </div>


          {/*
           * The statutory provision deliberately uses a
           * document/legal serif font rather than the normal
           * application interface font.
           *
           * 15px + 28px line height gives substantially more
           * information density than the previous 17px version.
           */}

          <div
            className="
              font-legal
              mt-4
              max-w-[78ch]
              whitespace-pre-wrap
              text-[15px]
              leading-7
              text-slate-700
            "
          >
            {section.description}
          </div>

        </main>


        {/* =====================================================
            RIGHT: MACHINE-DERIVED SMART INSIGHTS
        ====================================================== */}

        <aside
          className="
            border-l
            border-violet-100
            bg-violet-50/35
            px-4
            py-5
          "
        >

          <div
            className="
              flex
              items-center
              gap-2
              text-[11px]
              font-bold
              uppercase
              tracking-wider
              text-legal-purple
            "
          >
            <BrainCircuit size={15} />

            Smart Insights
          </div>


          {/* Insight loading */}

          {state.insightsLoading && (

            <div className="mt-5 space-y-3">

              {[1, 2, 3].map(
                (item) => (

                  <div
                    key={item}
                    className="
                      h-20
                      animate-pulse
                      rounded-lg
                      bg-violet-100/60
                    "
                  />

                ),
              )}

            </div>

          )}


          {/* Insight unavailable/degraded */}

          {!state.insightsLoading
            &&
            (
              state.insightsError
              ||
              insights?.partial_ml
            )
            &&
            (

              <div
                className="
                  mt-5
                  rounded-lg
                  border
                  border-amber-200
                  bg-amber-50
                  p-3
                "
              >

                <div
                  className="
                    text-xs
                    font-semibold
                    text-amber-800
                  "
                >
                  Smart Insights unavailable
                </div>


                <p
                  className="
                    mt-2
                    text-[11px]
                    leading-5
                    text-amber-700
                  "
                >
                  {
                    state.insightsError
                    ||
                    insights?.warnings.join(' ')
                    ||
                    'Graph intelligence is currently unavailable.'
                  }
                </p>


                <button
                  onClick={() =>
                    void state.retryInsights()
                  }
                  className="
                    mt-3
                    flex
                    items-center
                    gap-2
                    text-[11px]
                    font-bold
                    text-amber-800
                  "
                >
                  <RefreshCw size={12} />

                  Retry
                </button>

              </div>

            )}


          {/* Healthy Smart Insights */}

          {!state.insightsLoading
            &&
            !state.insightsError
            &&
            insights
            &&
            !insights.partial_ml
            &&
            (

              <div
                className="
                  mt-5
                  space-y-5
                "
              >

                {/* Key Concepts */}

                <InsightBlock
                  icon={
                    <Tags size={15} />
                  }
                  title="Key Concepts"
                >

                  {insights.concepts.length
                    ? (

                      <div
                        className="
                          flex
                          flex-wrap
                          gap-1.5
                        "
                      >

                        {insights.concepts.map(
                          (concept) => (

                            <span
                              key={concept}
                              className="
                                rounded-full
                                bg-white
                                px-2.5
                                py-1
                                text-[11px]
                                font-medium
                                text-legal-purple
                                ring-1
                                ring-violet-200
                              "
                            >
                              {concept}
                            </span>

                          ),
                        )}

                      </div>

                    )
                    : (

                      <Empty
                        text="No concepts identified yet."
                      />

                    )}

                </InsightBlock>


                {/* Related Sections */}

                <InsightBlock
                  icon={
                    <Network size={15} />
                  }
                  title="Related Provisions"
                >

                  {insights.related_sections.length
                    ? (

                      <div className="space-y-2">

                        {
                          insights.related_sections.map(
                            (item) => (

                              <button
                                key={
                                  item
                                    .section
                                    .section_number
                                }
                                onClick={() =>
                                  onOpenSection(
                                    item
                                      .section
                                      .section_number,
                                  )
                                }
                                className="
                                  w-full
                                  rounded-lg
                                  bg-white
                                  p-3
                                  text-left
                                  ring-1
                                  ring-violet-100
                                  hover:ring-violet-300
                                "
                              >

                                <div
                                  className="
                                    text-[10px]
                                    font-bold
                                    text-legal-gold
                                  "
                                >
                                  SECTION {
                                    item
                                      .section
                                      .section_number
                                  }
                                </div>


                                <div
                                  className="
                                    mt-1
                                    text-xs
                                    font-semibold
                                    leading-5
                                    text-legal-text
                                  "
                                >
                                  {
                                    item
                                      .section
                                      .section_name
                                  }
                                </div>


                                <div
                                  className="
                                    mt-2
                                    flex
                                    items-center
                                    gap-1
                                    text-[10px]
                                    font-semibold
                                    text-legal-purple
                                  "
                                >
                                  Open

                                  <ArrowRight
                                    size={11}
                                  />
                                </div>

                              </button>

                            ),
                          )
                        }

                      </div>

                    )
                    : (

                      <Empty
                        text="No related provisions available yet."
                      />

                    )}

                </InsightBlock>


                {/* Cross References */}

                <InsightBlock
                  icon={
                    <ExternalLink size={15} />
                  }
                  title="Cross References"
                >

                  {
                    insights
                      .cross_references
                      .length
                    ? (

                      <div
                        className="
                          flex
                          flex-wrap
                          gap-2
                        "
                      >

                        {
                          insights
                            .cross_references
                            .map(
                              (
                                reference,
                              ) => (

                                <button
                                  key={
                                    reference
                                  }
                                  onClick={() =>
                                    onOpenSection(
                                      reference,
                                    )
                                  }
                                  className="
                                    rounded-lg
                                    bg-white
                                    px-2.5
                                    py-1.5
                                    text-[11px]
                                    font-semibold
                                    text-legal-navy
                                    ring-1
                                    ring-slate-200
                                    hover:ring-legal-gold
                                  "
                                >
                                  Section {
                                    reference
                                  }
                                </button>

                              ),
                            )
                        }

                      </div>

                    )
                    : (

                      <Empty
                        text="No cross-references available."
                      />

                    )
                  }

                </InsightBlock>

              </div>

            )}

        </aside>

      </div>

    </div>
  );
}


function InsightBlock({
  icon,
  title,
  children,
}: {
  icon: ReactNode;
  title: string;
  children: ReactNode;
}) {

  return (
    <section>

      <div
        className="
          mb-2
          flex
          items-center
          gap-2
          text-sm
          font-semibold
          text-legal-text
        "
      >

        <span
          className="
            text-legal-purple
          "
        >
          {icon}
        </span>

        {title}

      </div>


      {children}

    </section>
  );
}


function Empty({
  text,
}: {
  text: string;
}) {

  return (
    <div
      className="
        rounded-lg
        border
        border-dashed
        border-violet-200
        bg-white/60
        px-3
        py-3
        text-[11px]
        leading-5
        text-legal-muted
      "
    >
      {text}
    </div>
  );
}