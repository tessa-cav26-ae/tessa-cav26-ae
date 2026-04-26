FROM nixos/nix

RUN mkdir -p /etc/nix && \
    echo "experimental-features = nix-command flakes" > /etc/nix/nix.conf && \
    echo "substituters = https://cache.nixos.org https://cuda-maintainers.cachix.org" >> /etc/nix/nix.conf && \
    echo "trusted-public-keys = cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY= cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E=" >> /etc/nix/nix.conf

# Copy the current directory into the image
COPY . /tessa
WORKDIR /tessa

RUN nix build .#storm --cores 8 --print-build-logs
RUN nix build .#stormpy --cores 8 --print-build-logs
RUN nix build .#tessa --cores 8 --print-build-logs


# ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en

CMD ["nix", "develop", "-c", "fish"]
