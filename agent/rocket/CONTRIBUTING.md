# Introduction

### Welcome !

First off, thank you for considering contributing to CityU Under Water Robotics. It's people like you that make Under Water Robotics such a great community.


### Why you should read the guidelines.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.


### What kinds of contributions we are looking for.

 Rocket is an internal project and we love to receive contributions from our community — you! There are many ways to contribute, from writing tutorials or blog posts, improving the documentation, submitting bug reports and feature requests or writing code which can be incorporated into Machine Learning itself.


### What contributions we are NOT looking for (if any).

Please, don't use the issue tracker for [support questions]. Check whether the #Features Slark channel on cityuur can help with your issue. If your problem is not strictly computer vision or machine learning specific, #python is generally more active. Stack Overflow is also worth considering.


# Ground Rules
### Behaviour we respect
 Responsibilities
 * Ensure cross-platform compatibility for every change that's accepted. Windows,   Ubuntu Linux.
 * Ensure that code that goes into core meets all requirements in this checklist: 
 * Create issues for any major changes and enhancements that you wish to make. Discuss things transparently and get community feedback.
 * Don't add any classes to the codebase unless absolutely needed. Err on the side of using functions.
 * Keep feature versions as small as possible, preferably one new feature per version.
 * Be welcoming to newcomers and encourage diverse new contributors from all backgrounds. See the [Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).


# Your First Contribution

 * Unsure where to begin contributing to Rocket? You can start by looking through these beginner and help-wanted issues:
 * Beginner issues - issues which should only require a few lines of code, and a test or two.
 * Help wanted issues - issues which should be a bit more involved than beginner issues.
 * Both issue lists are sorted by total number of comments. While not perfect, number of comments is a reasonable proxy for impact a given change will have.


Here are a couple of friendly tutorials included: 

### Working on your first Pull Request? You can learn how from this *free* series, [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github).



At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first :smile_cat:

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

# Getting started

* Get any other legal stuff out of the way
* How we run the test
* We currently using gitlab default issues

### For something that is bigger than a one or two line fix:

1. Create your own fork of the code
2. Do the changes in your fork
3. If you like the change and think the project could use it:
    * Be sure you have followed the code style for the project.
    * Send a pull request indicating that you have a changes on file.


### If you have a different process for small or "obvious" fixes

Small contributions such as fixing spelling errors, where the content is small enough to not be considered intellectual property, can be submitted by a contributor as a patch, without a CLA.

As a rule of thumb, changes are obvious fixes if they do not introduce any new functionality or creative thinking. As long as the change does not affect functionality, some likely examples include the following:
* Spelling / grammar fixes
* Typo correction, white space and formatting changes
* Comment clean up
* Bug fixes that change default return values or error codes stored in constants
* Adding logging messages or debugging output
* Changes to ‘metadata’ files like  .gitignore, build scripts, etc.
* Moving source files from one directory or package to another



# How to report a bug
### Explain security disclosures first!

If you find a security vulnerability, do NOT open an issue. whatsapp me +85296324677 or email me hottang4-c@my.cityu.edu.hk instead.


In order to determine whether you are dealing with a security issue, ask yourself these two questions:
* Can I access something that's not mine, or something I shouldn't have access to?
* Can I disable something for other people?
If the answer to either of those two questions are "yes", then you're probably dealing with a security issue. Note that even if you answer "no" to both questions, you may still be dealing with a security issue, so if you're unsure, just email us at security@travis-ci.org.


### Bug Report Template

When filing an issue, make sure to answer these five questions:
 1. What version of Go are you using (go version)?
 2. What operating system and processor architecture are you using?
 3. What did you do?
 4. What did you expect to see?
 5. What did you see instead?
General questions should go to the golang-nuts mailing list instead of the issue tracker. The gophers there will answer or ask you to file an issue if you've tripped over a bug.



# How to suggest a feature or enhancement
### If you have a particular roadmap, goals, or philosophy for development, share it here.
This information will give contributors context before they make suggestions that may not align with the project’s needs.
 Express does not force you to use any specific ORM or template engine. With support for over 14 template engines via Consolidate.js, you can quickly craft your perfect framework.


### Explain your desired process for suggesting a feature.

If you find yourself stucking for a feature that doesn't exist in Machine Learning, you are probably not alone. There are bound to be others out there with similar needs.
Many of the features that rocket has today have been added because we saw the need. Open an issue on our issues list on Gitlab which describes the feature you would like to see, why you need it, and how it should work.


# Code review process

The core team(only Me) looks at Pull Requests on a regular basis in a weekly triage meeting that we hold in a public Google Hangout. The hangout is announced in the weekly status updates that are sent to the puppet-dev list. Notes are posted to the Puppet Community community-triage repo and include a link to a YouTube recording of the hangout.
After feedback has been given we expect responses within two weeks. After two weeks we may close the pull request if it isn't showing any activity.


# Community

You can chat with the core team(me) on whatsapp. 


### Preferred style for code

use pylint to make sure basic style for python

### Commit message conventions.

clean, precise, specific to the changes

### Labeling conventions for issues.
Main branch
* master -> latest build
* dev/ -> development build
* \*.\*.\* -> features update

Tag
* release/v\*.\*.\* -> release tag
* v\*.\*.\* -> development tag
