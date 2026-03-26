export class UseAutoScroll {
	#ref = $state<HTMLElement>();
	#scrollY: number = $state(0);
	#userHasScrolled = $state(false);
	private lastScrollHeight = 0;

	set ref(ref: HTMLElement | undefined) {
		this.#ref = ref;

		if (!this.#ref) return;

		this.lastScrollHeight = this.#ref.scrollHeight;

		this.#ref.scrollTo(0, this.#scrollY ? this.#scrollY : this.#ref.scrollHeight);

		this.#ref.addEventListener('scroll', () => {
			if (!this.#ref) return;

			this.#scrollY = this.#ref.scrollTop;

			this.disableAutoScroll();
		});

		window.addEventListener('resize', () => {
			this.scrollToBottom(true);
		});

		const observer = new MutationObserver(() => {
			if (!this.#ref) return;

			if (this.#ref.scrollHeight !== this.lastScrollHeight) {
				this.scrollToBottom(true);
			}

			this.lastScrollHeight = this.#ref.scrollHeight;
		});

		observer.observe(this.#ref, { childList: true, subtree: true });
	}

	get ref() {
		return this.#ref;
	}

	get scrollY() {
		return this.#scrollY;
	}

	get isAtBottom() {
		if (!this.#ref) return true;

		return this.#scrollY + this.#ref.offsetHeight >= this.#ref.scrollHeight;
	}

	disableAutoScroll() {
		if (this.isAtBottom) {
			this.#userHasScrolled = false;
		} else {
			this.#userHasScrolled = true;
		}
	}

	scrollToBottom(auto = false) {
		if (!this.#ref) return;

		if (auto && this.#userHasScrolled) return;

		this.#ref.scrollTo(0, this.#ref.scrollHeight);
	}
}
