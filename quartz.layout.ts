import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/agioracle",
      Twitter: "https://twitter.com/agioracle",
      RSS: "/index.xml",
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs(),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(),
    Component.DesktopOnly(
      Component.RecentNotes({
        limit: 3,
        showTags: false,
        filter: (f) =>
          (f.slug!.startsWith("writings/explanation") || f.slug!.startsWith("writings/how-to") || f.slug!.startsWith("writings/reference") || f.slug!.startsWith("writings/tutorial") || f.slug!.startsWith("writings/fiction")) && f.slug! !== "writings/index" && !f.frontmatter?.noindex,
        linkToMore: "writings/" as SimpleSlug,
      }),
    ),
    Component.DesktopOnly(Component.Explorer()),
  ],
  right: [
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Graph({
      localGraph: {
        showTags: false,
      },
      globalGraph: {
        showTags: false,
      },
    }),
    Component.Backlinks(),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(),
    Component.DesktopOnly(Component.Explorer()),
  ],
  right: [],
}
