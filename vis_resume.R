# For a description of the VisualResume package, also see here: https://github.com/ndphillips/VisualResume

#install.packages("devtools") # Only if you don't have the devtools package
devtools::install_github("ndphillips/VisualResume")

VisualResume::VisualResume(
    titles.left = c("Dimitris Polychronopoulos", "Molecular Biology, Bioinformatics, Algorithms for Molecular Biology", "*Built with love in R using the InfoResume package: www.ndphillips.github.io/inforesume"),
    titles.right = c("https://dpolychr.github.io/", "dpolychr@gmail.com", "Twitter: @dpolychr2"),
    titles.right.cex = c(2, 2, 1),
    titles.left.cex = c(4, 2, 1),
    timeline.labels = c("Education", "Skills"),
    timeline = data.frame(title = c("DUTH", "BSRC Fleming", "CRUK Paterson Institute", "University of Athens", "NCSR Demokritos", "Military Duty", "Imperial College London", "Genomics England"),
                          sub = c("Ptychion", "Graduate Researcher", "Graduate Researcher", "MSc", "PhD", "Duty", "PostDoc", "Data Scientist"),
                          start = c(2002, 2006.9, 2008.1, 2008.10, 2011.6, 2015.3, 2015.9, 2017.8),
                          end = c(2006.5, 2007.11, 2008.9, 2011.2, 2014.9, 2015.8, 2017.7, 2018.10),
                          side = c(1, 1, 1, 1, 1, 0, 0, 0)),
    milestones = data.frame(title = c("BA", "MS", "PhD"),
                            sub = c("Molecular Biology", "Bioinformatics", "Bioinformatics"),
                            year = c(2006, 2011, 2014)),
    events = data.frame(year = c(2009, 2012, 2014, 2017, 2018),
                        title = c("Gambus A, van Deursen F, Polychronopoulos D et al (EMBO J.)",
                                  "EMBO Short Term Fellowship @ EPFL",
                                  "Best PhD Thesis across disciplines",
                                  "Polychronopoulos et al (NAR Review)",
                                  "Ayad et al (Bioinformatics)")),
    interests = list("programming" = c(rep("R", 10), rep("Python", 1)),
                     "tools" = c(rep("tidyverse", 30), rep("caret", 5), rep("ggplot2", 10), rep("flexdashbord", 5)),
                     "other skills" = c(rep("Proactive", 10), rep("Communicative", 5),rep("Team Player", 5), rep("Pragmatic", 30))),
    year.steps = 1
)
