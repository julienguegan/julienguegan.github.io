source "https://rubygems.org"

gem "github-pages", group: :jekyll_plugins

install_if -> { RUBY_PLATFORM =~ %r!mingw|mswin|java! } do
  gem "tzinfo", "~> 1.2"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.1.0" if Gem.win_platform?
gem "webrick"
 
# If you have any plugins, put them here
group :jekyll_plugins do
  gem "jekyll-paginate"
  gem "jekyll-sitemap"
  gem "jekyll-gist"
  gem "jekyll-feed"
  gem "jekyll-include-cache"
  gem "jekyll-algolia"
  gem "jekyll-polyglot"
  gem "jekyll-seo-tag"
end

gem "bigdecimal", "~> 4.0"
gem "erb", "~> 6.0"
gem "base64", "~> 0.2.0"
gem "mutex_m", "~> 0.3.0"
gem "faraday-retry", "~> 2.4"
gem "logger", "~> 1.6"
gem "csv", "~> 3.3"
gem "ostruct", "~> 0.6.3"
