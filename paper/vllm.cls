% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A new style for FAIR pre-prints
% Author: dlp@meta.com
% Version: 1.1
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Class declarations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{vllm}
\LoadClassWithOptions{article}
\makeatletter

% Preserve the more permissive float behavior from the previous paper style so
% large diagrams do not accumulate at the end of the document.
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.8}
\setcounter{topnumber}{4}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{6}

% Layout %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Layout margins: set in main.tex via \usepackage[margin=...]{geometry}.
% \RequirePackage[top=2.25cm, bottom=2.5cm, left=2.5cm, right=2.5cm, columnsep=0.65cm]{geometry}
\RequirePackage[expansion=false]{microtype}
\IfFileExists{placeins.sty}{\RequirePackage{placeins}}{\providecommand{\FloatBarrier}{}}
\IfFileExists{hyphenat.sty}{\RequirePackage{hyphenat}}{}
\RequirePackage{setspace}
\RequirePackage{parskip}
\RequirePackage[english]{babel}
\IfFileExists{lipsum.sty}{\RequirePackage{lipsum}}{}
\RequirePackage{etoolbox}

\DisableLigatures[f]{family=sf*}

% Graphics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{graphicx}
\RequirePackage{subcaption}

% Tables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{booktabs}
\IfFileExists{nicematrix.sty}{\RequirePackage{nicematrix}}{}
\RequirePackage{multirow}
\RequirePackage{bm}
\newcommand{\nm}[1]{#1}

% Colorful stuff %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{xcolor}
\definecolor{metablue}{HTML}{0064E0}
\definecolor{metafg}{HTML}{1C2B33}
\definecolor{metabg}{HTML}{F1F4F7}

% References %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{hyperref}
\hypersetup{
  colorlinks,
  linkcolor=metablue,
  citecolor=metablue,
  urlcolor=metablue
}


% \RequirePackage[round,authoryear]{natbib}
% \def\bibfont{\small}

% Section and caption format %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\IfFileExists{titlesec.sty}{
  \RequirePackage{titlesec}
  \titleformat*{\paragraph}{\itshape}
  \titleformat*{\section}{\Large\sffamily\bfseries}
  \titleformat*{\subsection}{\large\sffamily\bfseries}
  \titleformat*{\subsubsection}{\normalsize\sffamily\bfseries}
}{}
\DeclareTextFontCommand{\textbf}{\bfseries\sffamily}

\RequirePackage{caption}
\DeclareCaptionLabelSeparator{custom}{}
\DeclareCaptionFormat{custom}{{\sffamily\textbf{#1 #2}} #3}
\captionsetup{singlelinecheck=false,format=custom,labelsep=custom,font=small}
\captionsetup[sub]{singlelinecheck=true,format=custom,labelsep=custom,font=small}

% Meta custom fonts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% http://c.caignaert.free.fr/Install-ttf-Font.pdf

\RequirePackage[T1]{fontenc}
\IfFileExists{lmodern.sty}{\RequirePackage{lmodern}}{}

% \pdfmapline{+optimistic < assets/Optimistic.ttf <T1-WGL4.enc}
% \renewcommand{\sfdefault}{optimistic}

% Optional: install assets/Optimistic.ttf and uncomment shapes + map line below.
% \DeclareFontFamily{T1}{optimistic}{}
% \DeclareFontShape{T1}{optimistic}{m}{n}{<-> s * [0.88] assets/optimistic}{}
% \DeclareFontShape{T1}{optimistic}{b}{n}{<-> s * [0.88] assets/optimistic}{}
% \pdfmapline{+optimistic < assets/Optimistic.ttf <T1-WGL4.enc}
% \renewcommand{\sfdefault}{optimistic}
\IfFileExists{lmodern.sty}{}{\renewcommand{\sfdefault}{cmss}}

% Authors, affiliations, infos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand\addtolist[5][]{
  \begingroup
    \ifdefempty{#3}{\def\sep{}}{\def\sep{#5}}%
    \xdef#3{#3\sep #4{#1}{#2}}%
  \endgroup
}

\newcommand\authorlist{}
\newcommand\authorformat[2][]{{\sffamily \bfseries #2$^{#1}$}}
\providecommand{\@author}{}
\renewcommand{\author}[1]{%
  \gdef\@author{#1}%
  \def\authorlist{\par{\large\sffamily\bfseries\@author}}%
}

\newcommand{\affiliationlist}{}
\newcommand{\affiliation}[2][]{%
  \ifdefempty{\affiliationlist}
    {\gdef\affiliationlist{$^{#1}$#2}}
    {\g@addto@macro\affiliationlist{, $^{#1}$#2}}%
}

\newcommand\contributionlist{}
\protected\def\contributionformat#1#2{{\small $^{#1}$#2}}
\newcommand\contribution[2][]{\addtolist[#1]{#2}{\contributionlist}{\contributionformat}{, }}

\newcommand\metadatalist{}
\protected\def\metadataformat#1#2{{\small {\sffamily \bfseries #1:} #2}}
\newcommand\metadata[2][]{\addtolist[#1]{#2}{\metadatalist}{\metadataformat}{\par}}

\newcommand{\abstractlist}{}
\long\def\abstract#1{\def\abstractlist{{\color{metafg} #1}}}
\newcommand{\email}[1]{\href{mailto:#1}{\texttt{#1}}}

\providecommand{\@date}{}
\renewcommand\date[1]{%
  \gdef\@date{#1}%
  \def\metadatalist{\metadataformat{Date}{#1}}%
}
\newcommand\correspondence[1]{\metadata[Correspondence]{#1}}

\providecommand{\@title}{}
\newcommand{\@subtitle}{}
\renewcommand{\title}[1]{%
  \gdef\@title{#1}%
  \def\titlelist{{\huge\sffamily\bfseries #1}}%
}
\newcommand{\subtitle}[1]{%
  \gdef\@subtitle{#1}%
  \def\subtitlelist{{\Large #1}}%
}
\newcommand{\subtitlelist}{}

% Logo row in \mymaketitle (widths use \textwidth at \maketitle time, after geometry).
\newcommand{\vllmTitleMetaColumnWidth}{0.52\textwidth}
\newcommand{\vllmTitleLogoColumnWidth}{0.3\textwidth}
\newcommand{\vllmTitleLogoGraphicWidth}{2.5cm}
% Horizontal nudge: negative \hspace pulls logo left; positive pushes right.
\newcommand{\vllmTitleLogoHSpace}{\hspace{0pt}}
% Vertical nudge: negative \raisebox moves logo up; positive moves down.
\newcommand{\vllmTitleLogoRaise}{\raisebox{0pt}}

\newcommand{\beginappendix}{\appendix{\huge\sffamily Appendix\par}}

% Titlebox %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\mymaketitle}{%
  \begingroup
  \setlength{\fboxsep}{10pt}%
  \noindent\colorbox{metabg}{%
    \parbox{\dimexpr\linewidth-2\fboxsep\relax}{%
      \setlength{\parindent}{0cm}
      \setlength{\parskip}{0.1cm}

      \begin{center}
        {\huge\sffamily\bfseries \@title\par}
        \ifdefempty{\@subtitle}{}{\vspace{0.5em}{\Large \@subtitle\par}}
        \vspace{1.0em}
        {\normalsize \@author\par}
        \ifdefempty{\affiliationlist}{}{%
          \vspace{0.5em}%
          {\small \affiliationlist\par}%
        }%
      \end{center}

      \vspace{1.0em}

      \ifdefempty{\abstractlist}{}{%
        \noindent{\sffamily\bfseries Abstract. }\abstractlist\par
      }

      \noindent
      \begin{minipage}[t]{\vllmTitleMetaColumnWidth}
        \raggedright
        \ifdefempty{\metadatalist}{}{{\small \metadatalist}}
      \end{minipage}%
      \hfill
      \begin{minipage}[t]{\vllmTitleLogoColumnWidth}
        \raggedleft
        \vllmTitleLogoHSpace\vllmTitleLogoRaise{%
          \includegraphics[width=\vllmTitleLogoGraphicWidth]{artworks/vllm-sr-logo.dark.pdf}%
        }%
      \end{minipage}%
    }%
  }%
  \par
  \endgroup
  \FloatBarrier
}

\if@twocolumn%
\renewcommand{\maketitle}{
\twocolumn[%
  \mymaketitle
  \vskip 0.3cm
]
}%
\else%
% In single-column mode, starred floats should behave like normal floats.
\renewenvironment{figure*}{\@float{figure}}{\end@float}
\renewenvironment{table*}{\@float{table}}{\end@float}
\renewcommand{\maketitle}{
  \mymaketitle
}%
\fi
\makeatletter
